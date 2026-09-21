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
#include <gazebo/physics/ode/ODEJoint.hh>
#include <gazebo/physics/ode/ODELink.hh>
#include <gazebo/physics/ode/ODEPhysics.hh>
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
    if (coupling_group_) dJointGroupDestroy(coupling_group_);
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
        sdf, "activeStiffness", {25.0, 12.0, 12.0, 12.0});
    active_max_effort_ = ReadDoubles(
        sdf, "activeMaxEffort", {3.0, 1.80, 1.80, 1.80});
    mimic_source_index_ = ReadIndices(
        sdf, "mimicSourceIndex", {0, 1, 2, 3});

    const std::size_t active_size = active_names_.size();
    const std::size_t mimic_size = mimic_names_.size();
    if (active_size == 0 || mimic_size == 0 ||
        initial_targets_.size() != active_size ||
        active_stiffness_.size() != active_size ||
        active_max_effort_.size() != active_size ||
        mimic_source_index_.size() != mimic_size)
    {
      gzerr << "StableHandSpringPlugin received inconsistent vector sizes\n";
      return;
    }

    for (std::size_t index = 0; index < active_size; ++index)
    {
      physics::JointPtr joint = model_->GetJoint(active_names_[index]);
      if (!joint ||
          !ValidMotor(active_stiffness_[index], active_max_effort_[index]))
      {
        gzerr << "StableHandSpringPlugin invalid active joint "
              << active_names_[index] << "\n";
        return;
      }
      active_joints_.push_back(joint);
      position_mode_velocity_limits_.push_back(joint->GetVelocityLimit(0));
      lower_limits_.push_back(joint->LowerLimit(0));
      upper_limits_.push_back(joint->UpperLimit(0));
      initial_targets_[index] = Clamp(
          initial_targets_[index], lower_limits_.back(), upper_limits_.back());
    }
    for (std::size_t index = 0; index < mimic_size; ++index)
    {
      physics::JointPtr joint = model_->GetJoint(mimic_names_[index]);
      if (!joint || mimic_source_index_[index] >= active_size)
      {
        gzerr << "StableHandSpringPlugin invalid mimic joint "
              << mimic_names_[index] << "\n";
        return;
      }
      mimic_joints_.push_back(joint);
      mimic_lower_limits_.push_back(joint->LowerLimit(0));
      mimic_upper_limits_.push_back(joint->UpperLimit(0));
      // This joint has no motor or commanded spring equilibrium. Retain
      // only passive URDF damping; the transmission constraint carries load.
      joint->SetStiffnessDamping(0, 0.0, joint->GetDamping(0), 0.0);
    }

    // Initialize the complete coupled configuration while the model is being
    // loaded, before constraints and the first joint-state observation exist.
    // spawn_model applies its joint values after plugin construction. An
    // initial zero-configuration observation can otherwise be frozen by the
    // camera's pre-C hold as if it were the intended hand configuration.
    // This is a one-time spawn pose, never a motor-control or contact override.
    for (std::size_t index = 0; index < active_joints_.size(); ++index)
      active_joints_[index]->SetPosition(0, initial_targets_[index]);
    for (std::size_t index = 0; index < mimic_joints_.size(); ++index)
      mimic_joints_[index]->SetPosition(0, initial_targets_[mimic_source_index_[index]]);

    if (!CreatePassiveCouplings())
      return;

    targets_ = initial_targets_;
    shaft_positions_ = initial_targets_;
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
        "Physical grasp hand enabled: self-locking worm drives for "
        << active_size << " active and " << mimic_size
        << " passive 1:1 coupled joints; command topic " << command_topic);
  }

private:
  bool CreatePassiveCouplings()
  {
    if (world_->Physics()->GetType() != "ode")
    {
      gzerr << "The physical hand requires ODE transmission constraints\n";
      return false;
    }
    for (std::size_t index = 0; index < mimic_joints_.size(); ++index)
    {
      const auto source = active_joints_[mimic_source_index_[index]];
      const auto follower = mimic_joints_[index];
      const bool serial = follower->GetParent() == source->GetChild();
      const bool parallel = follower->GetParent() == source->GetParent();
      const double alignment = source->GlobalAxis(0).Dot(follower->GlobalAxis(0));
      const auto axis = source->GlobalAxis(0).Normalized();
      const auto separation = follower->Anchor(0) - source->Anchor(0);
      const double radius = 0.5 * (separation - axis * axis.Dot(separation)).Length();
      if ((!serial && !parallel) || (serial && alignment < 0.9999) ||
          (parallel && alignment > -0.9999) ||
          !std::isfinite(radius) || radius <= 1e-5)
      {
        gzerr << "Unsupported transmission geometry for " << mimic_names_[index] << "\n";
        return false;
      }
      coupling_serial_.push_back(serial);
    }
    coupling_group_ = dJointGroupCreate(0);
    return coupling_group_ != nullptr;
  }

  void UpdatePassiveCouplings()
  {
    // Equal-radius rolling gears give q_follower=q_source. For a serial
    // finger the reference gear is fixed to the carrier's parent, and the
    // distal gear orbits it: distal absolute rotation is twice the proximal
    // rotation. The parallel spread gears rotate oppositely in world axes.
    // A spatial point-velocity row includes both linear and angular terms,
    // so a moving carrier and its reaction are part of the SAME solve.
    // The native angular-only gearbox omitted that carrier contribution.
    dJointGroupEmpty(coupling_group_);
    const auto ode = boost::static_pointer_cast<physics::ODEPhysics>(world_->Physics());
    const double gain = coupling_erp_ / world_->Physics()->GetMaxStepSize();
    for (std::size_t index = 0; index < mimic_joints_.size(); ++index)
    {
      const auto source = active_joints_[mimic_source_index_[index]];
      const auto follower = mimic_joints_[index];
      const auto first = coupling_serial_[index] ? source->GetParent() : source->GetChild();
      const auto second = follower->GetChild();
      const auto axis = source->GlobalAxis(0).Normalized();
      const auto separation = follower->Anchor(0) - source->Anchor(0);
      const auto radial = separation - axis * axis.Dot(separation);
      const double radius = 0.5 * radial.Length();
      const auto point = 0.5 * (source->Anchor(0) + follower->Anchor(0));
      const auto tangent = axis.Cross(radial).Normalized();
      dContact constraint{};
      // Use one bilateral tangent row as an ideal transmission, NOT a
      // frictional object contact. The unused normal row is kept inactive;
      // no object attachment, follower motor, or SetPosition is involved.
      constraint.surface.mode = dContactMu2 | dContactFDir1 | dContactMotionN |
          dContactMotion1 | dContactSlip1 | dContactSlip2 | dContactSoftCFM;
      constraint.surface.mu = dInfinity;
      constraint.surface.mu2 = 0.0;
      constraint.surface.motionN = -100.0;
      constraint.surface.slip1 = constraint.surface.slip2 = 1e-8;
      constraint.surface.soft_cfm = 1e-8;
      const double error = follower->Position(0) - source->Position(0);
      constraint.surface.motion1 = (coupling_serial_[index] ? -1.0 : 1.0) * gain * radius * error;
      for (int coordinate = 0; coordinate < 3; ++coordinate)
      {
        constraint.geom.pos[coordinate] = point[coordinate];
        constraint.geom.normal[coordinate] = axis[coordinate];
        constraint.fdir1[coordinate] = tangent[coordinate];
      }
      auto joint = dJointCreateContact(ode->GetWorldId(), coupling_group_, &constraint);
      dJointAttach(joint,
          boost::static_pointer_cast<physics::ODELink>(first)->GetODEId(),
          boost::static_pointer_cast<physics::ODELink>(second)->GetODEId());
    }
  }

  struct PendingTrajectory
  {
    std::vector<std::vector<double>> points;
    std::vector<double> times;
    std::vector<double> motor_effort;
    bool valid = false;
  };

  static double Clamp(double value, double lower, double upper)
  {
    return std::max(lower, std::min(value, upper));
  }

  static bool ValidMotor(double stiffness, double maximum_effort)
  {
    return std::isfinite(stiffness) && stiffness > 0.0 &&
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
    if (message->joint_names.size() != active_names_.size())
    {
      ROS_ERROR_THROTTLE(
          1.0, "Physical hand rejected trajectory with wrong joint count");
      return;
    }
    std::unordered_map<std::string, std::size_t> message_indices;
    for (std::size_t index = 0; index < message->joint_names.size(); ++index)
    {
      if (!message_indices.emplace(message->joint_names[index], index).second)
      {
        ROS_ERROR_THROTTLE(
            1.0, "Physical hand rejected duplicate trajectory joint");
        return;
      }
    }
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
      std::vector<double> effort;
      if (!point.effort.empty())
      {
        if (point.effort.size() != active_names_.size())
          return;
        for (std::size_t i = 0; i < active_names_.size(); ++i)
        {
          const double value = point.effort[message_indices[active_names_[i]]];
          if (!std::isfinite(value) || value < 0.0 ||
              value > active_max_effort_[i] + 1e-9 || (i == 0 && value != 0.0))
          {
            ROS_ERROR_THROTTLE(1.0, "Physical hand rejected invalid motor torque");
            return;
          }
          effort.push_back(value);
        }
        if (effort.size() != 4 || std::fabs(effort[1] - effort[2]) > 1e-9 ||
            std::fabs(effort[1] - effort[3]) > 1e-9)
          return;
      }
      if (!pending.points.empty() && pending.motor_effort != effort)
        return;  // One synchronized motor mode for the whole command.
      pending.motor_effort = effort;
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
      shaft_positions_ = initial_targets_;
      motor_effort_.clear();
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
      const bool leaving_torque_mode = !motor_effort_.empty() && pending.motor_effort.empty();
      motor_effort_ = pending.motor_effort;
      trajectory_start_targets_ = targets_;
      if (leaving_torque_mode)
      {
        // Release starts at the measured contact pose, not at an unreachable
        // old CLOSE angle that would briefly reapply squeeze during opening.
        for (std::size_t i = 1; i < active_joints_.size(); ++i)
          trajectory_start_targets_[i] = Clamp(active_joints_[i]->Position(0),
                                               lower_limits_[i], upper_limits_[i]);
        targets_ = trajectory_start_targets_;
      }
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

    UpdatePassiveCouplings();
    // Torque mode applies the same motor effort at EVERY physics step,
    // independent of angle error, object contact, or arm trajectory.
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
      const auto joint = active_joints_[index];
      const double position = joint->Position(0);
      if (!std::isfinite(position))
        continue;
      const bool torque_mode = !motor_effort_.empty() && index > 0;
      if (force)
      {
        // Gazebo's implicit spring repurposes the ODE stops. The self-lock
        // brake owns these stops instead; no compliant motor spring remains.
        auto ode_joint = boost::dynamic_pointer_cast<physics::ODEJoint>(joint);
        ode_joint->UseImplicitSpringDamper(false);
        joint->SetStiffnessDamping(0, 0.0, 0.0, 0.0);
        joint->SetVelocityLimit(0, -1.0);
        joint->SetParam("stop_erp", 0, 0.8);
        joint->SetParam("stop_cfm", 0, 1.0e-9);
        joint->SetParam("bounce", 0, 0.0);
      }

      // The shaft is allowed to advance only while its motor is commanded
      // in that direction. A stopped drive locks in BOTH directions. Never
      // update its held angle from contact-induced backward displacement.
      double &shaft = shaft_positions_[index];
      const double requested = torque_mode ? motor_effort_[index] :
          targets_[index] - shaft;
      const int direction = requested > 1.0e-7 ? 1 : requested < -1.0e-7 ? -1 : 0;
      // One shaft drives the complete 1:1 transmission. Do not ratchet a
      // root-only solver overshoot into the held position while its passive
      // follower is still blocked: that would accumulate numerical strain.
      double achieved = position;
      for (std::size_t follower = 0; follower < mimic_joints_.size(); ++follower)
      {
        if (mimic_source_index_[follower] != index) continue;
        const double coupled = mimic_joints_[follower]->Position(0);
        if (direction > 0) achieved = std::min(achieved, coupled);
        if (direction < 0) achieved = std::max(achieved, coupled);
      }
      // A commanded shaft cannot advance faster than its original motor
      // speed. The moving stop only resists excess motion; it never pushes
      // the finger through an object to reach an imposed position.
      const double step = position_mode_velocity_limits_[index] *
          world_->Physics()->GetMaxStepSize();
      double low = shaft, high = shaft;
      if (direction > 0)
      {
        shaft = std::max(shaft, std::min({achieved, shaft + step,
            torque_mode ? upper_limits_[index] : targets_[index]}));
        low = shaft;
        high = std::min(shaft + step,
            torque_mode ? upper_limits_[index] : std::max(shaft, targets_[index]));
      }
      else if (direction < 0)
      {
        shaft = std::min(shaft, std::max({achieved, shaft - step,
            torque_mode ? lower_limits_[index] : targets_[index]}));
        low = std::max(shaft - step,
            torque_mode ? lower_limits_[index] : std::min(shaft, targets_[index]));
        high = shaft;
      }
      // Solver constraints transmit external reaction to the palm/arm.
      // This does not teleport links or command any passive distal motor.
      joint->SetUpperLimit(0, Clamp(high, lower_limits_[index], upper_limits_[index]));
      joint->SetLowerLimit(0, Clamp(low, lower_limits_[index], upper_limits_[index]));
      joint->SetUpperLimit(0, Clamp(high, lower_limits_[index], upper_limits_[index]));
      const double effort = torque_mode ? motor_effort_[index] : Clamp(
          active_stiffness_[index] * (targets_[index] - position),
          -active_max_effort_[index], active_max_effort_[index]);
      // Motor effort and passive self-lock reaction are distinct: the three
      // grip motors remain at 1.8 N m even when an outside load is larger.
      joint->SetForce(0, effort);
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
    stream << "{\"mode\":\""
           << (motor_effort_.empty() ? "SELF_LOCKING_POSITION" : "SELF_LOCKING_CONSTANT_TORQUE")
           << "\",\"motor_torque_command_nm\":[";
    for (std::size_t i = 1; i < active_joints_.size(); ++i)
      stream << (i > 1 ? "," : "") << (motor_effort_.empty() ? 0.0 : motor_effort_[i]);
    stream << "],\"motor_effort_readback_nm\":[";
    for (std::size_t i = 1; i < active_joints_.size(); ++i)
      stream << (i > 1 ? "," : "") << active_joints_[i]->GetForce(0);
    stream << "],\"sim_time_s\":"
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
  dJointGroupID coupling_group_ = nullptr;
  std::vector<bool> coupling_serial_;
  // Match the active shaft stop's correction strength without increasing
  // the global iteration count (which slows physical grasp contacts).
  double coupling_erp_ = 0.8;
  std::vector<double> lower_limits_;
  std::vector<double> upper_limits_;
  std::vector<double> mimic_lower_limits_;
  std::vector<double> mimic_upper_limits_;
  std::vector<double> initial_targets_;
  std::vector<double> targets_;
  std::vector<double> motor_effort_;
  std::vector<double> shaft_positions_;
  std::vector<double> last_published_positions_;
  std::vector<double> active_stiffness_;
  std::vector<double> active_max_effort_;
  std::vector<double> position_mode_velocity_limits_;
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
