#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/String.h>
#include <std_srvs/SetBool.h>

#include <moveit/collision_detection/collision_tools.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/joint_model.h>
#include <moveit/robot_model/link_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit_msgs/GetStateValidity.h>
#include <srdfdom/model.h>

namespace
{
const std::vector<std::string> kDefaultRequiredJointNames = {
  "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
  "joint_6", "f1j1", "f1j2", "f2j1", "f3j2",
};

const std::vector<std::string> kDefaultArmJointNames = {
  "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
};

std::string pairText(const collision_detection::CollisionResult& result)
{
  if (result.contacts.empty())
    return "unknown,unknown";
  const auto& names = result.contacts.begin()->first;
  return names.first + "," + names.second;
}
}  // namespace

class FullRobotSelfCollisionGuard
{
public:
  FullRobotSelfCollisionGuard()
    : private_nh_("~")
    , loader_(private_nh_.param<std::string>("robot_description", "robot_description"))
    , model_(loader_.getModel())
  {
    if (!model_)
      throw std::runtime_error("cannot load MoveIt robot model");

    private_nh_.param<std::string>("joint_state_topic", joint_state_topic_, "/joint_states");
    private_nh_.param<std::string>("service_name", service_name_,
                                   "/full_robot_self_collision_guard/check_state_validity");
    private_nh_.param("state_timeout_s", state_timeout_s_, 0.25);
    private_nh_.param("status_rate_hz", status_rate_hz_, 50.0);
    private_nh_.param<std::string>(
        "raw_velocity_topic", raw_velocity_topic_,
        "/full_robot_self_collision_guard/raw_arm_velocity");
    private_nh_.param<std::string>(
        "safe_velocity_topic", safe_velocity_topic_,
        "/abbarm_velocity_controller/command");
    private_nh_.param<std::string>(
        "hand_motion_service", hand_motion_service_,
        "/full_robot_self_collision_guard/set_hand_motion_active");
    private_nh_.param("command_timeout_s", command_timeout_s_, 0.10);
    private_nh_.param("prediction_horizon_s", prediction_horizon_s_, 0.20);
    private_nh_.param("maximum_prediction_step_rad",
                      maximum_prediction_step_rad_, 0.01);
    private_nh_.param("measured_joint_limit_tolerance_rad",
                      measured_joint_limit_tolerance_rad_, 0.0);
    if (!private_nh_.getParam("required_joint_names", required_joint_names_))
      required_joint_names_ = kDefaultRequiredJointNames;
    if (!private_nh_.getParam("arm_joint_names", arm_joint_names_))
      arm_joint_names_ = kDefaultArmJointNames;
    if (!(std::isfinite(state_timeout_s_) && state_timeout_s_ > 0.0 &&
          std::isfinite(status_rate_hz_) && status_rate_hz_ >= 20.0 &&
          std::isfinite(command_timeout_s_) && command_timeout_s_ > 0.0 &&
          std::isfinite(prediction_horizon_s_) && prediction_horizon_s_ >= 0.10 &&
          std::isfinite(maximum_prediction_step_rad_) &&
          maximum_prediction_step_rad_ > 0.0 &&
          maximum_prediction_step_rad_ <= 0.02 &&
          std::isfinite(measured_joint_limit_tolerance_rad_) &&
          measured_joint_limit_tolerance_rad_ >= 0.0 &&
          measured_joint_limit_tolerance_rad_ <= 0.02))
      throw std::runtime_error("invalid guard timing parameters");

    validateRequiredJoints();
    validateArmJoints();
    scene_.reset(new planning_scene::PlanningScene(model_));
    strict_acm_.reset(new collision_detection::AllowedCollisionMatrix(
        scene_->getAllowedCollisionMatrix()));
    configureStrictAllowedCollisionMatrix();

    latest_state_.reset(new moveit::core::RobotState(model_));
    latest_state_->setToDefaultValues();
    latest_state_->update();

    safe_publisher_ = nh_.advertise<std_msgs::Bool>(
        "/full_robot_self_collision_guard/safe", 1, true);
    diagnostic_publisher_ = nh_.advertise<std_msgs::String>(
        "/full_robot_self_collision_guard/diagnostic", 1, true);
    emergency_stop_publisher_ = nh_.advertise<std_msgs::Bool>(
        "/shared_teleop/emergency_stop", 1, true);
    safe_velocity_publisher_ = nh_.advertise<std_msgs::Float64MultiArray>(
        safe_velocity_topic_, 1, false);
    command_safe_publisher_ = nh_.advertise<std_msgs::Bool>(
        "/full_robot_self_collision_guard/command_safe", 1, true);
    command_diagnostic_publisher_ = nh_.advertise<std_msgs::String>(
        "/full_robot_self_collision_guard/command_diagnostic", 1, true);
    state_subscriber_ = nh_.subscribe(
        joint_state_topic_, 2, &FullRobotSelfCollisionGuard::stateCallback, this,
        ros::TransportHints().tcpNoDelay());
    raw_velocity_subscriber_ = nh_.subscribe(
        raw_velocity_topic_, 1,
        &FullRobotSelfCollisionGuard::rawVelocityCallback, this,
        ros::TransportHints().tcpNoDelay());
    service_ = nh_.advertiseService(
        service_name_, &FullRobotSelfCollisionGuard::checkCandidate, this);
    hand_motion_service_server_ = nh_.advertiseService(
        hand_motion_service_,
        &FullRobotSelfCollisionGuard::setHandMotionActive, this);
    status_timer_ = nh_.createTimer(
        ros::Duration(1.0 / status_rate_hz_),
        &FullRobotSelfCollisionGuard::statusTimer, this);

    publishStatus(false, "WAITING_FOR_COMPLETE_JOINT_STATE");
    publishCommandStatus(false, "WAITING_FOR_COMPLETE_JOINT_STATE");
    publishZeroVelocity();
    ROS_WARN_STREAM("Strict full-robot self-collision guard armed on "
                    << joint_state_topic_ << "; candidate service="
                    << service_name_ << "; MoveIt/FCL velocity gate="
                    << raw_velocity_topic_ << " -> " << safe_velocity_topic_);
  }

private:
  const moveit::core::LinkModel* nearestCollisionAncestor(
      const moveit::core::LinkModel* link) const
  {
    if (!link)
      return nullptr;
    const moveit::core::LinkModel* parent = link->getParentLinkModel();
    while (parent && parent->getShapes().empty())
      parent = parent->getParentLinkModel();
    return parent;
  }

  bool areAdjacentCollisionBodies(const moveit::core::LinkModel* first,
                                  const moveit::core::LinkModel* second) const
  {
    return first && second &&
        (nearestCollisionAncestor(first) == second ||
         nearestCollisionAncestor(second) == first);
  }

  void configureStrictAllowedCollisionMatrix()
  {
    const srdf::ModelConstSharedPtr& semantic = model_->getSRDF();
    if (!semantic)
      throw std::runtime_error("MoveIt semantic model is unavailable");

    std::size_t adjacent_count = 0;
    std::size_t reenabled_count = 0;
    for (const srdf::Model::CollisionPair& pair :
         semantic->getDisabledCollisionPairs())
    {
      const moveit::core::LinkModel* first =
          model_->getLinkModel(pair.link1_);
      const moveit::core::LinkModel* second =
          model_->getLinkModel(pair.link2_);
      if (!first || !second)
        throw std::runtime_error(
            "SRDF disabled collision pair references an unknown link: " +
            pair.link1_ + "," + pair.link2_);

      if (pair.reason_ == "Adjacent")
      {
        // Only consecutive collision bodies are allowed to remain exempt.
        // Fixed, geometry-free helper links (for example flange) are skipped
        // exactly as MoveIt's collision model skips them.
        if (!areAdjacentCollisionBodies(first, second))
          throw std::runtime_error(
              "SRDF Adjacent exemption is not kinematically adjacent: " +
              pair.link1_ + "," + pair.link2_);
        ++adjacent_count;
        continue;
      }
      if (pair.reason_ == "StructuralAdjacent")
      {
        // Servo uses one global proximity threshold, so a few normal assembly
        // clearances are excluded from proximity scaling. Binary safety checks
        // must still test them and therefore re-enable them here.
        strict_acm_->setEntry(pair.link1_, pair.link2_, false);
        ++reenabled_count;
        continue;
      }
      throw std::runtime_error(
          "unsafe SRDF collision exemption is forbidden: " + pair.link1_ +
          "," + pair.link2_ + " reason=" + pair.reason_);
    }
    ROS_INFO_STREAM("Strict MoveIt ACM loaded: " << adjacent_count
                    << " true adjacent exemptions; " << reenabled_count
                    << " structural proximity pairs re-enabled");
  }

  void validateRequiredJoints()
  {
    if (required_joint_names_.empty())
      throw std::runtime_error("required_joint_names cannot be empty");
    std::set<std::string> unique;
    for (const std::string& name : required_joint_names_)
    {
      if (!unique.insert(name).second)
        throw std::runtime_error("required_joint_names contains a duplicate");
      if (std::find(model_->getVariableNames().begin(),
                    model_->getVariableNames().end(), name) ==
          model_->getVariableNames().end())
        throw std::runtime_error("required joint variable is absent: " + name);
      const moveit::core::JointModel* joint = model_->getJointOfVariable(name);
      if (!joint || joint->getMimic())
        throw std::runtime_error("required joints must be independent variables: " + name);
    }
  }

  void validateArmJoints()
  {
    if (arm_joint_names_.size() != 6)
      throw std::runtime_error("arm_joint_names must contain exactly six joints");
    std::set<std::string> unique;
    for (const std::string& name : arm_joint_names_)
    {
      if (!unique.insert(name).second)
        throw std::runtime_error("arm_joint_names contains a duplicate");
      const auto& variables = model_->getVariableNames();
      if (std::find(variables.begin(), variables.end(), name) == variables.end())
        throw std::runtime_error("arm velocity joint is absent: " + name);
      const moveit::core::JointModel* joint = model_->getJointOfVariable(name);
      if (!joint || joint->getMimic())
        throw std::runtime_error("arm velocity joints must be independent: " + name);
    }
  }

  bool applyIndependentPositions(const sensor_msgs::JointState& message,
                                 moveit::core::RobotState& state,
                                 bool require_complete,
                                 double bounds_tolerance_rad,
                                 std::string& error) const
  {
    if (message.name.size() != message.position.size())
    {
      error = "joint name/position length mismatch";
      return false;
    }
    std::map<std::string, double> values;
    for (std::size_t index = 0; index < message.name.size(); ++index)
    {
      const std::string& name = message.name[index];
      const double value = message.position[index];
      if (!values.emplace(name, value).second)
      {
        error = "duplicate joint variable: " + name;
        return false;
      }
      if (!std::isfinite(value))
      {
        error = "non-finite joint position: " + name;
        return false;
      }
    }

    if (require_complete)
    {
      for (const std::string& name : required_joint_names_)
      {
        if (values.find(name) == values.end())
        {
          error = "incomplete joint state; missing " + name;
          return false;
        }
      }
    }

    bool applied = false;
    for (const auto& item : values)
    {
      if (std::find(model_->getVariableNames().begin(),
                    model_->getVariableNames().end(), item.first) ==
          model_->getVariableNames().end())
      {
        error = "unknown joint variable: " + item.first;
        return false;
      }
      const moveit::core::JointModel* joint = model_->getJointOfVariable(item.first);
      // A caller commands only independent variables. setVariablePosition()
      // then updates all corresponding mimic variables deterministically.
      if (!joint || joint->getMimic())
      {
        if (require_complete)
          continue;  // Ignore measured mimic joints from Gazebo.
        error = "candidate may not directly command mimic joint: " + item.first;
        return false;
      }
      state.setVariablePosition(item.first, item.second);
      applied = true;
    }
    if (!applied)
    {
      error = "no independent joint positions supplied";
      return false;
    }
    state.update();
    if (!state.satisfiesBounds(bounds_tolerance_rad))
    {
      std::ostringstream details;
      details << "joint limit violation";
      for (const moveit::core::JointModel* joint : model_->getActiveJointModels())
      {
        if (joint->satisfiesPositionBounds(
                state.getVariablePositions() + joint->getFirstVariableIndex(),
                bounds_tolerance_rad))
          continue;
        const std::vector<std::string>& names = joint->getVariableNames();
        const std::vector<moveit::core::VariableBounds>& bounds =
            joint->getVariableBounds();
        for (std::size_t index = 0; index < names.size(); ++index)
        {
          const double value = state.getVariablePosition(names[index]);
          if ((bounds[index].position_bounded_ &&
               (value < bounds[index].min_position_ - bounds_tolerance_rad ||
                value > bounds[index].max_position_ + bounds_tolerance_rad)))
          {
            details << ":" << names[index] << "=" << value
                    << " not_in[" << bounds[index].min_position_
                    << "," << bounds[index].max_position_ << "]";
          }
        }
      }
      error = details.str();
      return false;
    }
    return true;
  }

  bool collisionFree(moveit::core::RobotState& state,
                     collision_detection::CollisionResult& result) const
  {
    collision_detection::CollisionRequest request;
    request.contacts = true;
    request.max_contacts = 100;
    request.max_contacts_per_pair = 10;
    scene_->checkSelfCollision(request, result, state, *strict_acm_);
    return !result.collision;
  }

  void latchFault(const std::string& reason)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!fault_latched_)
    {
      fault_latched_ = true;
      fault_reason_ = reason;
      ROS_ERROR_STREAM("FULL-ROBOT SELF-COLLISION FAULT LATCHED: " << reason);
    }
  }

  void publishZeroVelocity()
  {
    std_msgs::Float64MultiArray zero;
    zero.data.assign(arm_joint_names_.size(), 0.0);
    safe_velocity_publisher_.publish(zero);
  }

  void publishCommandStatus(bool safe, const std::string& detail)
  {
    std_msgs::Bool status;
    status.data = safe;
    command_safe_publisher_.publish(status);
    std_msgs::String diagnostic;
    diagnostic.data = detail;
    command_diagnostic_publisher_.publish(diagnostic);
  }

  bool predictedVelocityIsSafe(const std::vector<double>& velocity,
                               moveit::core::RobotState& start,
                               std::string& reason) const
  {
    double maximum_travel = 0.0;
    for (double value : velocity)
      maximum_travel = std::max(maximum_travel,
                                std::abs(value) * prediction_horizon_s_);
    const std::size_t samples = std::max<std::size_t>(
        1, static_cast<std::size_t>(
               std::ceil(maximum_travel / maximum_prediction_step_rad_)));
    moveit::core::RobotState candidate(start);
    for (std::size_t sample = 1; sample <= samples; ++sample)
    {
      const double time = prediction_horizon_s_ *
          static_cast<double>(sample) / static_cast<double>(samples);
      candidate = start;
      for (std::size_t index = 0; index < arm_joint_names_.size(); ++index)
      {
        const std::string& name = arm_joint_names_[index];
        candidate.setVariablePosition(
            name, start.getVariablePosition(name) + velocity[index] * time);
      }
      candidate.update();
      if (!candidate.satisfiesBounds(0.0))
      {
        std::ostringstream stream;
        stream << "PREDICTED_JOINT_LIMIT:sample=" << sample << "/" << samples;
        reason = stream.str();
        return false;
      }
      collision_detection::CollisionResult result;
      if (!collisionFree(candidate, result))
      {
        std::ostringstream stream;
        stream << "PREDICTED_SELF_COLLISION:" << pairText(result)
               << ":sample=" << sample << "/" << samples;
        reason = stream.str();
        return false;
      }
    }
    reason = "SAFE";
    return true;
  }

  void rawVelocityCallback(const std_msgs::Float64MultiArrayConstPtr& message)
  {
    if (message->data.size() != arm_joint_names_.size())
    {
      publishZeroVelocity();
      publishCommandStatus(false, "MALFORMED_VELOCITY_COMMAND_SIZE");
      ROS_ERROR_STREAM_THROTTLE(
          1.0, "MoveIt velocity gate rejected command with "
          << message->data.size() << " values; expected "
          << arm_joint_names_.size());
      return;
    }
    std::vector<double> velocity(message->data.begin(), message->data.end());
    if (!std::all_of(velocity.begin(), velocity.end(),
                     [](double value) { return std::isfinite(value); }))
    {
      publishZeroVelocity();
      publishCommandStatus(false, "NONFINITE_VELOCITY_COMMAND");
      ROS_ERROR_STREAM_THROTTLE(
          1.0, "MoveIt velocity gate rejected a non-finite command");
      return;
    }

    moveit::core::RobotState start(model_);
    bool ready = false;
    bool hand_motion_active = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const double state_age = have_complete_state_ ?
          (ros::WallTime::now() - latest_state_wall_time_).toSec() :
          std::numeric_limits<double>::infinity();
      ready = have_complete_state_ && !fault_latched_ &&
              state_age <= state_timeout_s_;
      hand_motion_active = hand_motion_active_;
      if (ready)
        start = *latest_state_;
      last_raw_command_wall_time_ = ros::WallTime::now();
      have_raw_command_ = true;
    }
    if (!ready)
    {
      publishZeroVelocity();
      publishCommandStatus(false, "CURRENT_STATE_UNSAFE_OR_STALE");
      return;
    }
    if (hand_motion_active)
    {
      publishZeroVelocity();
      publishCommandStatus(false, "HAND_MOTION_INTERLOCK_ACTIVE");
      return;
    }

    std::string reason;
    if (!predictedVelocityIsSafe(velocity, start, reason))
    {
      publishZeroVelocity();
      publishCommandStatus(false, reason);
      ROS_WARN_STREAM_THROTTLE(
          0.5, "MoveIt/FCL blocked predicted arm command: " << reason);
      return;
    }
    std_msgs::Float64MultiArray safe = *message;
    safe_velocity_publisher_.publish(safe);
    publishCommandStatus(true, "SAFE");
  }

  bool setHandMotionActive(std_srvs::SetBool::Request& request,
                           std_srvs::SetBool::Response& response)
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      hand_motion_active_ = request.data;
    }
    if (request.data)
      publishZeroVelocity();
    response.success = true;
    response.message = request.data ?
        "arm velocity interlocked for collision-checked hand motion" :
        "arm velocity interlock released";
    ROS_WARN_STREAM("Full-robot motion interlock: hand_motion_active="
                    << (request.data ? "true" : "false"));
    return true;
  }

  void stateCallback(const sensor_msgs::JointStateConstPtr& message)
  {
    // /joint_states may be split across several publishers.  The physical
    // grasp hand, for example, publishes its eight joints independently from
    // gazebo_ros_control's six arm joints.  Validate each fragment first, then
    // merge only independent variables into a short-lived snapshot.  A
    // complete state is accepted only while every required fragment is fresh;
    // statusTimer still latches a fault if either publisher disappears.
    if (message->name.size() != message->position.size())
    {
      latchFault("INVALID_JOINT_STATE:joint name/position length mismatch");
      return;
    }
    std::map<std::string, double> updates;
    for (std::size_t index = 0; index < message->name.size(); ++index)
    {
      const std::string& name = message->name[index];
      const double value = message->position[index];
      if (!updates.emplace(name, value).second)
      {
        latchFault("INVALID_JOINT_STATE:duplicate joint variable: " + name);
        return;
      }
      if (!std::isfinite(value))
      {
        latchFault("INVALID_JOINT_STATE:non-finite joint position: " + name);
        return;
      }
      if (std::find(model_->getVariableNames().begin(),
                    model_->getVariableNames().end(), name) ==
          model_->getVariableNames().end())
      {
        latchFault("INVALID_JOINT_STATE:unknown joint variable: " + name);
        return;
      }
    }

    const ros::WallTime received_at = ros::WallTime::now();
    sensor_msgs::JointState merged;
    moveit::core::RobotState candidate(model_);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      candidate = *latest_state_;
      for (const auto& item : updates)
      {
        const moveit::core::JointModel* joint =
            model_->getJointOfVariable(item.first);
        if (!joint || joint->getMimic())
          continue;
        latest_independent_positions_[item.first] = item.second;
        latest_joint_wall_times_[item.first] = received_at;
      }
      for (const std::string& name : required_joint_names_)
      {
        const auto position = latest_independent_positions_.find(name);
        const auto stamp = latest_joint_wall_times_.find(name);
        if (position == latest_independent_positions_.end() ||
            stamp == latest_joint_wall_times_.end() ||
            (received_at - stamp->second).toSec() > state_timeout_s_)
          return;
        merged.name.push_back(name);
        merged.position.push_back(position->second);
      }
    }
    std::string error;
    if (!applyIndependentPositions(
            merged, candidate, true, measured_joint_limit_tolerance_rad_, error))
    {
      latchFault("INVALID_JOINT_STATE:" + error);
      return;
    }

    collision_detection::CollisionResult result;
    if (!collisionFree(candidate, result))
    {
      latchFault("SELF_COLLISION:" + pairText(result));
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      *latest_state_ = candidate;
      latest_state_wall_time_ = received_at;
      have_complete_state_ = true;
    }
  }

  bool checkCandidate(moveit_msgs::GetStateValidity::Request& request,
                      moveit_msgs::GetStateValidity::Response& response)
  {
    moveit::core::RobotState candidate(model_);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const double age = have_complete_state_ ?
          (ros::WallTime::now() - latest_state_wall_time_).toSec() :
          std::numeric_limits<double>::infinity();
      if (fault_latched_ || !have_complete_state_ || age > state_timeout_s_)
      {
        response.valid = false;
        return true;
      }
      candidate = *latest_state_;
    }

    const sensor_msgs::JointState& overrides = request.robot_state.joint_state;
    if (!overrides.name.empty() || !overrides.position.empty())
    {
      std::string error;
      if (!applyIndependentPositions(overrides, candidate, false, 0.0, error))
      {
        ROS_WARN_STREAM_THROTTLE(1.0, "Rejected strict collision query: " << error);
        response.valid = false;
        return true;
      }
    }

    collision_detection::CollisionResult result;
    response.valid = collisionFree(candidate, result);
    if (!response.valid)
    {
      for (const auto& contact_pair : result.contacts)
      {
        for (const collision_detection::Contact& contact : contact_pair.second)
        {
          moveit_msgs::ContactInformation message;
          collision_detection::contactToMsg(contact, message);
          message.header.frame_id = model_->getModelFrame();
          message.header.stamp = ros::Time::now();
          response.contacts.push_back(message);
        }
      }
    }
    return true;
  }

  void publishStatus(bool safe, const std::string& detail)
  {
    std_msgs::Bool status;
    status.data = safe;
    safe_publisher_.publish(status);
    std_msgs::String diagnostic;
    diagnostic.data = detail;
    diagnostic_publisher_.publish(diagnostic);
  }

  void statusTimer(const ros::TimerEvent&)
  {
    bool have_state;
    bool fault;
    double age;
    std::string detail;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      have_state = have_complete_state_;
      fault = fault_latched_;
      detail = fault_reason_;
      age = have_state ?
          (ros::WallTime::now() - latest_state_wall_time_).toSec() :
          std::numeric_limits<double>::infinity();
    }
    if (!fault && have_state && age > state_timeout_s_)
    {
      latchFault("JOINT_STATE_TIMEOUT");
      fault = true;
      detail = "JOINT_STATE_TIMEOUT";
    }
    const bool safe = have_state && !fault && age <= state_timeout_s_;
    publishStatus(safe, safe ? "SAFE" :
        (detail.empty() ? "WAITING_FOR_COMPLETE_JOINT_STATE" : detail));
    if (!safe && have_state)
    {
      std_msgs::Bool stop;
      stop.data = true;
      emergency_stop_publisher_.publish(stop);
    }

    bool command_stale = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      command_stale = have_raw_command_ &&
          (ros::WallTime::now() - last_raw_command_wall_time_).toSec() >
              command_timeout_s_;
    }
    if (!safe || command_stale)
    {
      publishZeroVelocity();
      if (command_stale)
        publishCommandStatus(false, "RAW_VELOCITY_COMMAND_TIMEOUT");
    }
  }

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  robot_model_loader::RobotModelLoader loader_;
  moveit::core::RobotModelPtr model_;
  planning_scene::PlanningScenePtr scene_;
  std::unique_ptr<collision_detection::AllowedCollisionMatrix> strict_acm_;
  moveit::core::RobotStatePtr latest_state_;
  std::mutex mutex_;
  std::map<std::string, double> latest_independent_positions_;
  std::map<std::string, ros::WallTime> latest_joint_wall_times_;
  ros::WallTime latest_state_wall_time_;
  bool have_complete_state_{ false };
  bool fault_latched_{ false };
  std::string fault_reason_;
  std::string joint_state_topic_;
  std::string service_name_;
  std::string raw_velocity_topic_;
  std::string safe_velocity_topic_;
  std::string hand_motion_service_;
  std::vector<std::string> required_joint_names_;
  std::vector<std::string> arm_joint_names_;
  double state_timeout_s_{ 0.25 };
  double status_rate_hz_{ 50.0 };
  double command_timeout_s_{ 0.10 };
  double prediction_horizon_s_{ 0.20 };
  double maximum_prediction_step_rad_{ 0.01 };
  double measured_joint_limit_tolerance_rad_{ 0.0 };
  ros::WallTime last_raw_command_wall_time_;
  bool have_raw_command_{ false };
  bool hand_motion_active_{ false };
  ros::Publisher safe_publisher_;
  ros::Publisher diagnostic_publisher_;
  ros::Publisher emergency_stop_publisher_;
  ros::Publisher safe_velocity_publisher_;
  ros::Publisher command_safe_publisher_;
  ros::Publisher command_diagnostic_publisher_;
  ros::Subscriber state_subscriber_;
  ros::Subscriber raw_velocity_subscriber_;
  ros::ServiceServer service_;
  ros::ServiceServer hand_motion_service_server_;
  ros::Timer status_timer_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "full_robot_self_collision_guard");
  try
  {
    FullRobotSelfCollisionGuard guard;
    ros::spin();
  }
  catch (const std::exception& error)
  {
    ROS_FATAL_STREAM("Full-robot self-collision guard failed: " << error.what());
    return 8;
  }
  return 0;
}
