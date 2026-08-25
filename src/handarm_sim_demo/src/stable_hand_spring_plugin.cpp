#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <boost/bind/bind.hpp>
#include <gazebo/common/Events.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/String.h>
#include <trajectory_msgs/JointTrajectory.h>

namespace gazebo
{
class StableHandSpringPlugin : public ModelPlugin
{
public:
  StableHandSpringPlugin() = default;

  ~StableHandSpringPlugin() override
  {
    alive_.store(false);
    queue_.disable();
    command_subscriber_.shutdown();
    if (queue_thread_.joinable())
      queue_thread_.join();
    update_connection_.reset();
  }

  void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override
  {
    model_ = model;
    world_ = model_->GetWorld();
    if (!ros::isInitialized())
    {
      gzerr << "StableHandSpringPlugin requires gazebo_ros_api_plugin\n";
      return;
    }

    active_names_ = ReadStrings(
        sdf, "activeJointNames", {"f1j1", "f1j2", "f2j1", "f3j2"});
    mimic_names_ = ReadStrings(
        sdf, "mimicJointNames", {"f3j1", "f1j3", "f2j2", "f3j3"});
    initial_targets_ = ReadDoubles(
        sdf, "initialPositions", {0.051, 0.0317, 0.0227, 0.0363});
    active_stiffness_ = ReadDoubles(
        sdf, "activeStiffness", {20.0, 12.0, 12.0, 12.0});
    active_damping_ = ReadDoubles(
        sdf, "activeDamping", {0.14, 0.08, 0.08, 0.08});
    active_max_effort_ = ReadDoubles(
        sdf, "activeMaxEffort", {3.0, 0.60, 0.60, 0.60});
    mimic_stiffness_ = ReadDoubles(
        sdf, "mimicStiffness", {20.0, 10.0, 10.0, 10.0});
    mimic_damping_ = ReadDoubles(
        sdf, "mimicDamping", {0.14, 0.06, 0.06, 0.06});
    mimic_max_effort_ = ReadDoubles(
        sdf, "mimicMaxEffort", {3.0, 0.40, 0.40, 0.40});
    mimic_source_index_ = ReadIndices(
        sdf, "mimicSourceIndex", {0, 1, 2, 3});

    const std::size_t active_size = active_names_.size();
    const std::size_t mimic_size = mimic_names_.size();
    if (active_size == 0 || mimic_size == 0 ||
        initial_targets_.size() != active_size ||
        active_stiffness_.size() != active_size ||
        active_damping_.size() != active_size ||
        active_max_effort_.size() != active_size ||
        mimic_stiffness_.size() != mimic_size ||
        mimic_damping_.size() != mimic_size ||
        mimic_max_effort_.size() != mimic_size ||
        mimic_source_index_.size() != mimic_size)
    {
      gzerr << "StableHandSpringPlugin received inconsistent vector sizes\n";
      return;
    }

    for (std::size_t index = 0; index < active_size; ++index)
    {
      physics::JointPtr joint = model_->GetJoint(active_names_[index]);
      if (!joint ||
          !ValidSpring(
              active_stiffness_[index], active_damping_[index],
              active_max_effort_[index]))
      {
        gzerr << "StableHandSpringPlugin invalid active joint "
              << active_names_[index] << "\n";
        return;
      }
      active_joints_.push_back(joint);
      lower_limits_.push_back(joint->LowerLimit(0));
      upper_limits_.push_back(joint->UpperLimit(0));
      initial_targets_[index] = Clamp(
          initial_targets_[index], lower_limits_.back(), upper_limits_.back());
    }
    for (std::size_t index = 0; index < mimic_size; ++index)
    {
      physics::JointPtr joint = model_->GetJoint(mimic_names_[index]);
      if (!joint || mimic_source_index_[index] >= active_size ||
          !ValidSpring(
              mimic_stiffness_[index], mimic_damping_[index],
              mimic_max_effort_[index]))
      {
        gzerr << "StableHandSpringPlugin invalid mimic joint "
              << mimic_names_[index] << "\n";
        return;
      }
      mimic_joints_.push_back(joint);
      mimic_lower_limits_.push_back(joint->LowerLimit(0));
      mimic_upper_limits_.push_back(joint->UpperLimit(0));
    }

    targets_ = initial_targets_;
    applied_active_references_.assign(
        active_size, std::numeric_limits<double>::quiet_NaN());
    applied_mimic_references_.assign(
        mimic_size, std::numeric_limits<double>::quiet_NaN());
    last_published_positions_.assign(
        active_size + mimic_size,
        std::numeric_limits<double>::quiet_NaN());
    ApplyReferences(true);

    const std::string robot_namespace = ReadString(sdf, "robotNamespace", "/");
    const std::string command_topic = ReadString(
        sdf, "commandTopic", "/controller_gazebo_hand/command");
    const std::string state_topic = ReadString(sdf, "stateTopic", "/joint_states");
    const std::string diagnostic_topic = ReadString(
        sdf, "diagnosticTopic", "/handarm_sim_demo/physical_hand_diagnostics");
    publish_rate_hz_ = ReadDouble(sdf, "publishRate", 50.0);
    if (!std::isfinite(publish_rate_hz_) || publish_rate_hz_ <= 0.0)
    {
      gzerr << "StableHandSpringPlugin publishRate must be positive\n";
      return;
    }

    node_.reset(new ros::NodeHandle(robot_namespace));
    ros::SubscribeOptions options =
        ros::SubscribeOptions::create<trajectory_msgs::JointTrajectory>(
            command_topic, 1,
            boost::bind(
                &StableHandSpringPlugin::OnCommand, this,
                boost::placeholders::_1),
            ros::VoidPtr(), &queue_);
    command_subscriber_ = node_->subscribe(options);
    state_publisher_ = node_->advertise<sensor_msgs::JointState>(
        state_topic, 10, false);
    diagnostic_publisher_ = node_->advertise<std_msgs::String>(
        diagnostic_topic, 10, false);
    update_connection_ = event::Events::ConnectWorldUpdateBegin(
        std::bind(&StableHandSpringPlugin::OnUpdate, this));
    last_sim_time_s_ = world_->SimTime().Double();
    last_publish_time_s_ = last_sim_time_s_;
    alive_.store(true);
    queue_thread_ = std::thread(&StableHandSpringPlugin::QueueThread, this);

    ROS_WARN_STREAM(
        "Physical grasp hand enabled: implicit spring-damper control for "
        << active_size << " active and " << mimic_size
        << " coupled joints; command topic " << command_topic);
  }

private:
  struct PendingTrajectory
  {
    std::vector<std::vector<double>> points;
    std::vector<double> times;
    bool valid = false;
  };

  static double Clamp(double value, double lower, double upper)
  {
    return std::max(lower, std::min(value, upper));
  }

  static bool ValidSpring(
      double stiffness, double damping, double maximum_effort)
  {
    return std::isfinite(stiffness) && stiffness > 0.0 &&
           std::isfinite(damping) && damping > 0.0 &&
           std::isfinite(maximum_effort) && maximum_effort > 0.0;
  }

  static std::string ReadString(
      const sdf::ElementPtr &sdf, const std::string &name,
      const std::string &fallback)
  {
    return sdf->HasElement(name)
               ? sdf->GetElement(name)->Get<std::string>()
               : fallback;
  }

  static double ReadDouble(
      const sdf::ElementPtr &sdf, const std::string &name, double fallback)
  {
    return sdf->HasElement(name)
               ? sdf->GetElement(name)->Get<double>()
               : fallback;
  }

  static std::vector<std::string> ParseStrings(const std::string &text)
  {
    std::istringstream stream(text);
    std::vector<std::string> values;
    std::string value;
    while (stream >> value)
      values.push_back(value);
    return values;
  }

  static std::vector<double> ParseDoubles(const std::string &text)
  {
    std::istringstream stream(text);
    std::vector<double> values;
    double value = 0.0;
    while (stream >> value)
      values.push_back(value);
    return values;
  }

  static std::vector<std::size_t> ParseIndices(const std::string &text)
  {
    std::istringstream stream(text);
    std::vector<std::size_t> values;
    std::size_t value = 0;
    while (stream >> value)
      values.push_back(value);
    return values;
  }

  static std::vector<std::string> ReadStrings(
      const sdf::ElementPtr &sdf, const std::string &name,
      const std::vector<std::string> &fallback)
  {
    return sdf->HasElement(name)
               ? ParseStrings(sdf->GetElement(name)->Get<std::string>())
               : fallback;
  }

  static std::vector<double> ReadDoubles(
      const sdf::ElementPtr &sdf, const std::string &name,
      const std::vector<double> &fallback)
  {
    return sdf->HasElement(name)
               ? ParseDoubles(sdf->GetElement(name)->Get<std::string>())
               : fallback;
  }

  static std::vector<std::size_t> ReadIndices(
      const sdf::ElementPtr &sdf, const std::string &name,
      const std::vector<std::size_t> &fallback)
  {
    return sdf->HasElement(name)
               ? ParseIndices(sdf->GetElement(name)->Get<std::string>())
               : fallback;
  }

  static double SmoothStep(double value)
  {
    const double u = Clamp(value, 0.0, 1.0);
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
  }

  void OnCommand(const trajectory_msgs::JointTrajectoryConstPtr &message)
  {
    std::unordered_map<std::string, std::size_t> message_indices;
    for (std::size_t index = 0; index < message->joint_names.size(); ++index)
      message_indices[message->joint_names[index]] = index;
    for (const std::string &name : active_names_)
    {
      if (message_indices.find(name) == message_indices.end())
      {
        ROS_ERROR_THROTTLE(
            1.0, "Physical hand rejected trajectory missing joint %s",
            name.c_str());
        return;
      }
    }
    if (message->points.empty())
    {
      ROS_ERROR_THROTTLE(1.0, "Physical hand rejected empty trajectory");
      return;
    }

    PendingTrajectory pending;
    double previous_time = 0.0;
    for (const trajectory_msgs::JointTrajectoryPoint &point : message->points)
    {
      const double point_time = point.time_from_start.toSec();
      if (!std::isfinite(point_time) || point_time <= previous_time)
      {
        ROS_ERROR_THROTTLE(
            1.0, "Physical hand trajectory times must increase from zero");
        return;
      }
      std::vector<double> target(active_names_.size(), 0.0);
      for (std::size_t index = 0; index < active_names_.size(); ++index)
      {
        const std::size_t source = message_indices[active_names_[index]];
        if (source >= point.positions.size() ||
            !std::isfinite(point.positions[source]))
        {
          ROS_ERROR_THROTTLE(
              1.0, "Physical hand trajectory has invalid positions");
          return;
        }
        target[index] = Clamp(
            point.positions[source], lower_limits_[index], upper_limits_[index]);
      }
      pending.points.push_back(target);
      pending.times.push_back(point_time);
      previous_time = point_time;
    }
    pending.valid = true;
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      pending_trajectory_ = pending;
    }
  }

  void OnUpdate()
  {
    const double now = world_->SimTime().Double();
    if (now < last_sim_time_s_)
    {
      trajectory_active_ = false;
      targets_ = initial_targets_;
      std::fill(
          last_published_positions_.begin(), last_published_positions_.end(),
          std::numeric_limits<double>::quiet_NaN());
      last_publish_time_s_ = now;
      ApplyReferences(true);
    }
    last_sim_time_s_ = now;

    PendingTrajectory pending;
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      if (pending_trajectory_.valid)
      {
        pending = pending_trajectory_;
        pending_trajectory_.valid = false;
      }
    }
    if (pending.valid)
    {
      trajectory_start_targets_ = targets_;
      trajectory_points_ = pending.points;
      trajectory_times_ = pending.times;
      trajectory_start_time_s_ = now;
      trajectory_active_ = true;
    }

    if (trajectory_active_)
    {
      const double elapsed = std::max(0.0, now - trajectory_start_time_s_);
      std::size_t segment = 0;
      while (segment < trajectory_times_.size() &&
             elapsed > trajectory_times_[segment])
        ++segment;
      if (segment >= trajectory_times_.size())
      {
        targets_ = trajectory_points_.back();
        trajectory_active_ = false;
      }
      else
      {
        const std::vector<double> &left =
            segment == 0 ? trajectory_start_targets_ : trajectory_points_[segment - 1];
        const std::vector<double> &right = trajectory_points_[segment];
        const double left_time = segment == 0 ? 0.0 : trajectory_times_[segment - 1];
        const double duration = trajectory_times_[segment] - left_time;
        const double blend = SmoothStep((elapsed - left_time) / duration);
        for (std::size_t index = 0; index < targets_.size(); ++index)
          targets_[index] = left[index] + (right[index] - left[index]) * blend;
      }
    }

    // Recompute force-limited spring references every physics step. When a
    // finger is blocked by an object this caps spring torque instead of
    // allowing target error to build an arbitrarily large contact impulse.
    ApplyReferences(false);

    if (now - last_publish_time_s_ >= 1.0 / publish_rate_hz_)
    {
      PublishState(now);
      last_publish_time_s_ = now;
    }
  }

  void ApplyReferences(bool force)
  {
    for (std::size_t index = 0; index < active_joints_.size(); ++index)
    {
      const double position = active_joints_[index]->Position(0);
      if (!std::isfinite(position))
        continue;
      const double maximum_error =
          active_max_effort_[index] / active_stiffness_[index];
      const double reference = Clamp(
          position + Clamp(
              targets_[index] - position, -maximum_error, maximum_error),
          lower_limits_[index], upper_limits_[index]);
      if (force || !std::isfinite(applied_active_references_[index]) ||
          std::fabs(reference - applied_active_references_[index]) > 1.0e-7)
      {
        active_joints_[index]->SetStiffnessDamping(
            0, active_stiffness_[index], active_damping_[index], reference);
        applied_active_references_[index] = reference;
      }
    }
    for (std::size_t index = 0; index < mimic_joints_.size(); ++index)
    {
      const std::size_t source = mimic_source_index_[index];
      const double target = active_joints_[source]->Position(0);
      const double position = mimic_joints_[index]->Position(0);
      if (!std::isfinite(target) || !std::isfinite(position))
        continue;
      const double maximum_error =
          mimic_max_effort_[index] / mimic_stiffness_[index];
      const double reference = Clamp(
          position + Clamp(target - position, -maximum_error, maximum_error),
          mimic_lower_limits_[index], mimic_upper_limits_[index]);
      if (force || !std::isfinite(applied_mimic_references_[index]) ||
          std::fabs(reference - applied_mimic_references_[index]) > 1.0e-7)
      {
        mimic_joints_[index]->SetStiffnessDamping(
            0, mimic_stiffness_[index], mimic_damping_[index], reference);
        applied_mimic_references_[index] = reference;
      }
    }
  }

  void PublishState(double now)
  {
    sensor_msgs::JointState state;
    ros::Time stamp;
    stamp.fromSec(now);
    state.header.stamp = stamp;
    state.name = active_names_;
    state.name.insert(state.name.end(), mimic_names_.begin(), mimic_names_.end());
    state.position.reserve(active_joints_.size() + mimic_joints_.size());
    state.velocity.reserve(active_joints_.size() + mimic_joints_.size());
    state.effort.reserve(active_joints_.size() + mimic_joints_.size());
    const double sample_dt = now - last_publish_time_s_;
    std::vector<double> current_positions;
    current_positions.reserve(active_joints_.size() + mimic_joints_.size());
    double maximum_active_velocity = 0.0;
    double maximum_active_error = 0.0;
    for (std::size_t index = 0; index < active_joints_.size(); ++index)
    {
      const double position = active_joints_[index]->Position(0);
      const double previous = last_published_positions_[index];
      const double velocity =
          sample_dt > 0.0 && std::isfinite(previous)
              ? (position - previous) / sample_dt
              : 0.0;
      current_positions.push_back(position);
      state.position.push_back(position);
      state.velocity.push_back(velocity);
      state.effort.push_back(active_joints_[index]->GetForce(0));
      maximum_active_velocity = std::max(
          maximum_active_velocity, std::fabs(velocity));
      maximum_active_error = std::max(
          maximum_active_error, std::fabs(targets_[index] - position));
    }
    double maximum_mimic_velocity = 0.0;
    double maximum_mimic_error = 0.0;
    for (std::size_t index = 0; index < mimic_joints_.size(); ++index)
    {
      const double position = mimic_joints_[index]->Position(0);
      const std::size_t state_index = active_joints_.size() + index;
      const double previous = last_published_positions_[state_index];
      const double velocity =
          sample_dt > 0.0 && std::isfinite(previous)
              ? (position - previous) / sample_dt
              : 0.0;
      const double target =
          active_joints_[mimic_source_index_[index]]->Position(0);
      current_positions.push_back(position);
      state.position.push_back(position);
      state.velocity.push_back(velocity);
      state.effort.push_back(mimic_joints_[index]->GetForce(0));
      maximum_mimic_velocity = std::max(
          maximum_mimic_velocity, std::fabs(velocity));
      maximum_mimic_error = std::max(
          maximum_mimic_error, std::fabs(target - position));
    }
    last_published_positions_ = current_positions;
    state_publisher_.publish(state);

    std::ostringstream stream;
    stream << "{\"mode\":\"PHYSICAL_GRASP_SPRING\",\"sim_time_s\":"
           << now << ",\"trajectory_active\":"
           << (trajectory_active_ ? "true" : "false")
           << ",\"maximum_active_velocity_rad_s\":"
           << maximum_active_velocity
           << ",\"maximum_active_error_rad\":" << maximum_active_error
           << ",\"maximum_mimic_velocity_rad_s\":"
           << maximum_mimic_velocity
           << ",\"maximum_mimic_error_rad\":" << maximum_mimic_error
           << "}";
    std_msgs::String diagnostic;
    diagnostic.data = stream.str();
    diagnostic_publisher_.publish(diagnostic);
  }

  void QueueThread()
  {
    while (alive_.load() && ros::ok())
      queue_.callAvailable(ros::WallDuration(0.01));
  }

  physics::ModelPtr model_;
  physics::WorldPtr world_;
  std::vector<std::string> active_names_;
  std::vector<std::string> mimic_names_;
  std::vector<physics::JointPtr> active_joints_;
  std::vector<physics::JointPtr> mimic_joints_;
  std::vector<double> lower_limits_;
  std::vector<double> upper_limits_;
  std::vector<double> mimic_lower_limits_;
  std::vector<double> mimic_upper_limits_;
  std::vector<double> initial_targets_;
  std::vector<double> targets_;
  std::vector<double> applied_active_references_;
  std::vector<double> applied_mimic_references_;
  std::vector<double> last_published_positions_;
  std::vector<double> active_stiffness_;
  std::vector<double> active_damping_;
  std::vector<double> active_max_effort_;
  std::vector<double> mimic_stiffness_;
  std::vector<double> mimic_damping_;
  std::vector<double> mimic_max_effort_;
  std::vector<std::size_t> mimic_source_index_;
  std::vector<double> trajectory_start_targets_;
  std::vector<std::vector<double>> trajectory_points_;
  std::vector<double> trajectory_times_;
  double trajectory_start_time_s_ = 0.0;
  bool trajectory_active_ = false;
  double publish_rate_hz_ = 50.0;
  double last_sim_time_s_ = 0.0;
  double last_publish_time_s_ = 0.0;

  std::mutex command_mutex_;
  PendingTrajectory pending_trajectory_;
  std::unique_ptr<ros::NodeHandle> node_;
  ros::CallbackQueue queue_;
  ros::Subscriber command_subscriber_;
  ros::Publisher state_publisher_;
  ros::Publisher diagnostic_publisher_;
  event::ConnectionPtr update_connection_;
  std::atomic<bool> alive_{false};
  std::thread queue_thread_;
};

GZ_REGISTER_MODEL_PLUGIN(StableHandSpringPlugin)
}  // namespace gazebo
