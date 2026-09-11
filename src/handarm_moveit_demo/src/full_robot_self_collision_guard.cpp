#include <algorithm>
#include <array>
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
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>
#include <std_srvs/SetBool.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <handarm_moveit_demo/HandCommand.h>
#include <moveit_msgs/PlanningScene.h>
#include <moveit_msgs/GetPlanningScene.h>
#include <Eigen/SVD>
#include <qpOASES.hpp>

#include <geometric_shapes/shapes.h>
#include <moveit/collision_detection_fcl/collision_common.h>
#include <fcl/narrowphase/distance.h>
#include <moveit/collision_detection/collision_tools.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/joint_model.h>
#include <moveit/robot_model/link_model.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit_msgs/GetStateValidity.h>
#include <srdfdom/model.h>

namespace
{
// FCL 0.6's sphere/triangle routine returns each point in its shape frame.
// Query the specific shape pairs so their transforms and ordering are known.
// This avoids six complete-scene finite-difference queries per control tick.
bool robotPairWorldPoints(const moveit::core::RobotState& state,
                          const collision_detection::DistanceResultsData& contact,
                          Eigen::Vector3d* points)
{
  const auto model = state.getRobotModel();
  if (!model->hasLinkModel(contact.link_names[0]) || !model->hasLinkModel(contact.link_names[1]))
    return false;
  const auto* a = model->getLinkModel(contact.link_names[0]);
  const auto* b = model->getLinkModel(contact.link_names[1]);
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < a->getShapes().size(); ++i)
    for (std::size_t j = 0; j < b->getShapes().size(); ++j)
    {
      const auto ga = collision_detection::createCollisionGeometry(a->getShapes()[i], a, i);
      const auto gb = collision_detection::createCollisionGeometry(b->getShapes()[j], b, j);
      if (!ga || !gb) return false;
      const auto& ta = state.getCollisionBodyTransform(a, i);
      const auto& tb = state.getCollisionBodyTransform(b, j);
      fcl::DistanceResultd result;
      fcl::distance(ga->collision_geometry_.get(), ta, gb->collision_geometry_.get(), tb,
                    fcl::DistanceRequestd(true), result);
      if (!std::isfinite(result.min_distance) || result.min_distance < 0.0) return false;
      if (result.min_distance >= best) continue;
      const bool first_is_a = result.o1 == ga->collision_geometry_.get();
      if (!first_is_a && result.o1 != gb->collision_geometry_.get()) return false;
      Eigen::Vector3d pa = result.nearest_points[first_is_a ? 0 : 1];
      Eigen::Vector3d pb = result.nearest_points[first_is_a ? 1 : 0];
      const bool sphere_mesh =
          (a->getShapes()[i]->type == shapes::SPHERE && b->getShapes()[j]->type == shapes::MESH) ||
          (b->getShapes()[j]->type == shapes::SPHERE && a->getShapes()[i]->type == shapes::MESH);
      if (sphere_mesh)
      {
        pa = ta * pa;
        pb = tb * pb;
      }
      if (!pa.allFinite() || !pb.allFinite() ||
          std::abs((pa-pb).norm() - result.min_distance) > 1e-5) return false;
      points[0] = pa; points[1] = pb; best = result.min_distance;
    }
  return std::isfinite(best) && std::abs(best - contact.distance) <= 1e-5;
}

const std::vector<std::string> kDefaultRequiredJointNames = {
  "joint_1", "joint_2", "joint_3", "joint_4", "joint_5",
  "joint_6", "f1j1", "f1j2", "f2j1", "f3j2",
};

const std::vector<std::string> kDefaultArmJointNames = {
  "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
};

const std::vector<std::string> kDefaultHandReferenceJointNames = {
  "f1j1", "f1j2", "f2j1", "f3j2",
};

const std::vector<double> kDefaultHandReferencePositions = {
  0.051, 0.0317, 0.0227, 0.0363,
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
    private_nh_.param("inverse_damping", inverse_damping_, 0.0);
    private_nh_.param<std::string>(
        "raw_velocity_topic", raw_velocity_topic_,
        "/full_robot_self_collision_guard/raw_arm_velocity");
    private_nh_.param<std::string>(
        "safe_velocity_topic", safe_velocity_topic_,
        "/abbarm_velocity_controller/command");
    private_nh_.param<std::string>(
        "hand_motion_service", hand_motion_service_,
        "/full_robot_self_collision_guard/set_hand_motion_active");
    private_nh_.param<std::string>(
        "hand_command_topic", hand_command_topic_,
        "/controller_gazebo_hand/command");
    private_nh_.param("command_timeout_s", command_timeout_s_, 0.10);
    private_nh_.param(
        "enable_swept_command_gate", enable_swept_command_gate_, true);
    // The EGM emulation is a position-reference plant, so zeroing qdot does
    // not stop it instantaneously.  Keep enough look-ahead for reference
    // re-anchoring plus closed-loop braking under the configured speed cap.
    private_nh_.param("prediction_horizon_s", prediction_horizon_s_, 0.40);
    private_nh_.param("maximum_prediction_step_rad",
                      maximum_prediction_step_rad_, 0.01);
    private_nh_.param("maximum_prediction_samples",
                      maximum_prediction_samples_, 256);
    private_nh_.param("measured_joint_limit_tolerance_rad",
                      measured_joint_limit_tolerance_rad_, 0.0);
    private_nh_.param("recover_simulated_hand_limit_transients",
                      recover_simulated_hand_limit_transients_, false);
    if (!private_nh_.getParam("required_joint_names", required_joint_names_))
      required_joint_names_ = kDefaultRequiredJointNames;
    if (!private_nh_.getParam("arm_joint_names", arm_joint_names_))
      arm_joint_names_ = kDefaultArmJointNames;
    if (!private_nh_.getParam(
            "hand_reference_joint_names", hand_reference_joint_names_))
      hand_reference_joint_names_ = kDefaultHandReferenceJointNames;
    if (!private_nh_.getParam(
            "initial_hand_reference", initial_hand_reference_))
      initial_hand_reference_ = kDefaultHandReferencePositions;
    if (!(std::isfinite(state_timeout_s_) && state_timeout_s_ > 0.0 &&
          std::isfinite(status_rate_hz_) && status_rate_hz_ >= 20.0 &&
          std::isfinite(command_timeout_s_) && command_timeout_s_ > 0.0 &&
          std::isfinite(prediction_horizon_s_) && prediction_horizon_s_ >= 0.02 &&
          std::isfinite(maximum_prediction_step_rad_) &&
          maximum_prediction_step_rad_ > 0.0 &&
          maximum_prediction_step_rad_ <= 0.02 &&
          maximum_prediction_samples_ >= 1 &&
          maximum_prediction_samples_ <= 256 &&
          std::isfinite(measured_joint_limit_tolerance_rad_) &&
          measured_joint_limit_tolerance_rad_ >= 0.0 &&
          measured_joint_limit_tolerance_rad_ <= 0.02))
      throw std::runtime_error("invalid guard timing parameters");

    validateRequiredJoints();
    validateArmJoints();
    validateHandReference();
    scene_.reset(new planning_scene::PlanningScene(model_));
    strict_acm_.reset(new collision_detection::AllowedCollisionMatrix(
        scene_->getAllowedCollisionMatrix()));
    configureStrictAllowedCollisionMatrix();

    latest_state_.reset(new moveit::core::RobotState(model_));
    latest_state_->setToDefaultValues();
    latest_state_->update();

    private_nh_.param("geometric_command_filter", geometric_command_filter_, false);
    if (geometric_command_filter_)
    {
      nh_.param("/shared_teleop/control/constrained_velocity_projection",
                constrained_velocity_projection_, false);
      double side_degrees = 90.0;
      nh_.param("/shared_teleop/mapping/maximum_side_rotation_deg", side_degrees, 90.0);
      maximum_side_angle_ = side_degrees * M_PI / 180.0;
      if (!std::isfinite(maximum_side_angle_) || maximum_side_angle_ <= 0.0 ||
          maximum_side_angle_ > M_PI)
        throw std::runtime_error("invalid operator side-angle bound");
      robot_zero_subscriber_ = nh_.subscribe<geometry_msgs::PoseStamped>(
          "/shared_teleop/robot_zero_pose", 1,
          [this](const geometry_msgs::PoseStampedConstPtr& message) {
            if (message->header.frame_id != "base_link") return;
            const auto& q = message->pose.orientation;
            Eigen::Quaterniond rotation(q.w, q.x, q.y, q.z);
            if (!rotation.coeffs().allFinite() || rotation.norm() < 1e-6) return;
            robot_zero_rotation_ = rotation.normalized().toRotationMatrix();
            have_robot_zero_ = true;
          });
      bool simulation = false;
      nh_.param("/use_sim_time", simulation, false);
      if (constrained_velocity_projection_ && !simulation)
        throw std::runtime_error("constrained velocity projection is Gazebo-only");
      nh_.param("/shared_teleop/control/collision_guard_enter_scale",
                proximity_enter_scale_, 0.20);
      nh_.param("/shared_teleop/control/collision_guard_release_scale",
                proximity_release_scale_, 0.80);
      nh_.param("/servo_server/lower_singularity_threshold", singularity_soft_, 17.0);
      nh_.param("/servo_server/hard_stop_singularity_threshold", singularity_hard_, 30.0);
      nh_.param("/servo_server/joint_limit_margin", joint_limit_margin_, 0.08);
      proximity_scale_subscriber_ = nh_.subscribe<std_msgs::Float64>(
          "/servo_server/internal/collision_velocity_scale", 1,
          [this](const std_msgs::Float64ConstPtr& message) {
            if (!std::isfinite(message->data)) return;
            collision_scale_ = std::max(0.0, std::min(1.0, message->data));
            if (message->data <= proximity_enter_scale_)
              proximity_recovery_active_ = true;
            else if (message->data >= proximity_release_scale_)
              proximity_recovery_active_ = false;
          });
      planning_scene_client_ = nh_.serviceClient<moveit_msgs::GetPlanningScene>(
          "/get_planning_scene");
      geometric_command_publisher_ = nh_.advertise<handarm_moveit_demo::HandCommand>(
          "/shared_teleop/geometrically_limited_command", 1);
      geometric_diagnostic_publisher_ = nh_.advertise<std_msgs::String>(
          "/shared_teleop/geometric_filter_diagnostics", 1);
      geometric_command_subscriber_ = nh_.subscribe(
          "/shared_teleop/shaped_command", 1,
          &FullRobotSelfCollisionGuard::filterCartesianCommand, this);
      planning_scene_subscriber_ = nh_.subscribe<moveit_msgs::PlanningScene>(
          "/move_group/monitored_planning_scene", 1,
          [this](const moveit_msgs::PlanningSceneConstPtr& message) {
            if (!message->is_diff || have_scene_snapshot_)
              have_scene_snapshot_ = scene_->usePlanningSceneMsg(*message);
          });
    }

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
    hand_command_subscriber_ = nh_.subscribe(
        hand_command_topic_, 1,
        &FullRobotSelfCollisionGuard::handCommandCallback, this,
        ros::TransportHints().tcpNoDelay());
    const auto arm_reference_topic = private_nh_.param<std::string>(
        "arm_position_reference_topic", "");
    if (!arm_reference_topic.empty())
    {
      std::vector<double> accelerations;
      if (nh_.getParam("/egm_position_reference_adapter/maximum_acceleration", accelerations) &&
          accelerations.size() == arm_joint_names_.size())
        for (std::size_t i = 0; i < accelerations.size(); ++i)
          if (std::isfinite(accelerations[i]) && accelerations[i] > 0.0)
            reference_acceleration_[arm_joint_names_[i]] = accelerations[i];
      arm_reference_subscriber_ = nh_.subscribe<std_msgs::Float64MultiArray>(
          arm_reference_topic, 1,
          [this](const std_msgs::Float64MultiArrayConstPtr& message) {
            if ((message->data.size() != arm_joint_names_.size() &&
                 message->data.size() != 2 * arm_joint_names_.size()) ||
                !std::all_of(message->data.begin(), message->data.end(),
                             [](double x) { return std::isfinite(x); })) return;
            std::lock_guard<std::mutex> lock(mutex_);
            for (std::size_t i = 0; i < arm_joint_names_.size(); ++i)
            {
              latest_arm_reference_[arm_joint_names_[i]] = message->data[i];
              if (message->data.size() == 2 * arm_joint_names_.size())
                latest_reference_velocity_[arm_joint_names_[i]] = message->data[i + arm_joint_names_.size()];
            }
          });
    }
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
                    << raw_velocity_topic_ << " -> " << safe_velocity_topic_
                    << "; swept command gate="
                    << (enable_swept_command_gate_ ? "enabled" :
                        "disabled (MoveIt Servo is authoritative)"));
  }

private:
  struct Clearance
  {
    double self;
    double world;
  };

  double armMargin(const moveit::core::RobotState& state) const
  {
    double result = std::numeric_limits<double>::max();
    for (const auto& name : arm_joint_names_)
    {
      const auto& bounds = model_->getVariableBounds(name);
      if (!bounds.position_bounded_) continue;
      const double q = state.getVariablePosition(name);
      result = std::min(result, std::min(q - bounds.min_position_, bounds.max_position_ - q));
    }
    return result;
  }

  Clearance clearance(const moveit::core::RobotState& state) const
  {
    collision_detection::DistanceRequest request;
    request.group_name = "abbarm";
    // Without signed distances FCL reports the same -1 sentinel for every
    // penetrating pose. That hides an increasing-clearance retreat from a
    // shallow contact and can leave Servo's zero collision scale latched.
    // Keep the existing non-worsening test, using actual penetration depth.
    request.enable_signed_distance = true;
    request.enableGroup(model_);
    request.acm = &scene_->getAllowedCollisionMatrix();
    collision_detection::DistanceResult self, world;
    scene_->getCollisionEnv()->distanceSelf(request, self, state);
    scene_->getCollisionEnv()->distanceRobot(request, world, state);
    return { self.minimum_distance.distance, world.minimum_distance.distance };
  }

  void preserveMeasuredHandDeflection(const moveit::core::RobotState& measured,
                                     moveit::core::RobotState& predicted) const
  {
    // The Gazebo adaptive hand reports passive joints that deflect under
    // contact. Moving an arm joint, or predicting a small hand increment,
    // must not replace that observed shape by an instant rigid-mimic shape.
    for (const auto* joint : model_->getMimicJointModels())
    {
      const auto* source = joint->getMimic();
      if (joint->getVariableCount() != 1 || source->getVariableCount() != 1) continue;
      const auto& name = joint->getVariableNames()[0];
      const auto& source_name = source->getVariableNames()[0];
      predicted.setVariablePosition(name, measured.getVariablePosition(name) +
          joint->getMimicFactor() * (predicted.getVariablePosition(source_name) -
                                   measured.getVariablePosition(source_name)));
    }
  }

  bool robotWorldContact(const std::string& first, const std::string& second) const
  {
    // Existing world contact may separate; it must not disable every escape
    // direction. The normal and swept-depth checks still reject pressing in.
    // Self contact is never classified here or permitted by this exception.
    return (model_->hasLinkModel(first) && scene_->getWorld()->hasObject(second)) ||
           (model_->hasLinkModel(second) && scene_->getWorld()->hasObject(first));
  }

  collision_detection::CollisionResult worldContacts(moveit::core::RobotState& state) const
  {
    state.updateCollisionBodyTransforms();
    collision_detection::CollisionRequest request;
    request.group_name = "abbarm";
    request.contacts = true;
    request.max_contacts = 200;
    request.max_contacts_per_pair = 20;
    collision_detection::CollisionResult result;
    scene_->getCollisionEnv()->checkRobotCollision(request, result, state,
        scene_->getAllowedCollisionMatrix());
    return result;
  }

  bool projectJointVelocity(moveit::core::RobotState& state,
                            const Eigen::MatrixXd& jacobian,
                            const Eigen::Matrix<double, 6, 1>& requested,
                            std::vector<double>& velocity,
                            std::size_t& constraint_count,
                            std::string& reason) const
  {
    // Differential IK with velocity-damper inequalities, as used in NEO and
    // constrained differential IK. Unlike the legacy clearance conjunction,
    // a still-distant object can be approached while escaping a close wrist.
    // This is a local velocity controller, not a global obstacle planner.
    const auto* group = model_->getJointModelGroup("abbarm");
    const int n = static_cast<int>(arm_joint_names_.size());
    std::vector<Eigen::RowVectorXd> rows;
    std::vector<double> lower;
    state.updateCollisionBodyTransforms();
    collision_detection::DistanceRequest request;
    request.group_name = "abbarm";
    request.enableGroup(model_);
    request.enable_nearest_points = true;
    request.enable_signed_distance = true;
    request.type = collision_detection::DistanceRequestType::SINGLE;
    request.distance_threshold = 0.06;
    const double approach_rate = 5.0;
    // FCL's signed-distance nearest points can coincide inside a contact.
    // Use the actual contact normal (body 1 toward body 2) there. Its
    // derivative was checked against the recorded 09:23 fingertip contact.
    // This does not allow penetration: the swept check below must establish
    // that every existing finger/object contact gets shallower, with no new
    // contact. World contacts must separate; self contacts remain stops.
    const auto contacts = worldContacts(state);
    if (contacts.contact_count >= 200 || (contacts.collision && contacts.contacts.empty()))
    {
      reason = "WORLD_CONTACT_DETAILS_UNAVAILABLE";
      return false;
    }
    for (const auto& pair : contacts.contacts)
    {
      if (!robotWorldContact(pair.first.first, pair.first.second))
      {
        reason = "CURRENT_CONTACT:" + pair.first.first + "," + pair.first.second;
        return false;
      }
      for (const auto& contact : pair.second)
      {
        if (!contact.normal.allFinite() || contact.normal.norm() < 1e-9 ||
            !contact.pos.allFinite())
        {
          reason = "COLLISION_NORMAL_UNAVAILABLE";
          return false;
        }
        const bool first_is_link = contact.body_type_1 == collision_detection::BodyType::ROBOT_LINK;
        const auto* link = model_->getLinkModel(first_is_link ? contact.body_name_1 : contact.body_name_2);
        Eigen::MatrixXd point_jacobian;
        if (!link || !state.getJacobian(group, link,
            state.getGlobalLinkTransform(link).inverse() * contact.pos, point_jacobian))
        {
          reason = "COLLISION_JACOBIAN_UNAVAILABLE";
          return false;
        }
        rows.push_back((first_is_link ? -1.0 : 1.0) * contact.normal.normalized().transpose() *
                       point_jacobian.topRows(3));
        lower.push_back(0.0);  // Retain only directions that do not press farther in.
      }
    }
    for (bool self : { true, false })
    {
      request.acm = self ? strict_acm_.get() : &scene_->getAllowedCollisionMatrix();
      collision_detection::DistanceResult distances;
      if (self) scene_->getCollisionEnv()->distanceSelf(request, distances, state);
      else scene_->getCollisionEnv()->distanceRobot(request, distances, state);
      std::vector<collision_detection::DistanceResult> perturbed_distances;
      const double derivative_step = 1e-5;
      auto has_shape = [&](const std::string& name, shapes::ShapeType type) {
        if (model_->hasLinkModel(name))
        {
          for (const auto& shape : model_->getLinkModel(name)->getShapes())
            if (shape->type == type) return true;
        }
        else if (const auto object = scene_->getWorld()->getObject(name))
          for (const auto& shape : object->shapes_)
            if (shape->type == type) return true;
        return false;
      };
      for (const auto& pair : distances.distances)
      {
        for (const auto& contact : pair.second)
        {
          if (!std::isfinite(contact.distance) || contact.distance > request.distance_threshold) continue;
          if (contact.distance < 0.0)
          {
            if (!self && robotWorldContact(contact.link_names[0], contact.link_names[1]) &&
                contacts.contacts.count(std::minmax(contact.link_names[0], contact.link_names[1])))
              continue;  // Constrained by the contact normal above, then checked over the full step.
            reason = "CURRENT_CONTACT:" + contact.link_names[0] + "," + contact.link_names[1];
            return false;
          }
          Eigen::RowVectorXd gradient = Eigen::RowVectorXd::Zero(n);
          Eigen::Vector3d world_points[2] = {contact.nearest_points[0], contact.nearest_points[1]};
          Eigen::Vector3d separation = world_points[0] - world_points[1];
          const bool mesh_sphere =
              (has_shape(contact.link_names[0], shapes::MESH) && has_shape(contact.link_names[1], shapes::SPHERE)) ||
              (has_shape(contact.link_names[1], shapes::MESH) && has_shape(contact.link_names[0], shapes::SPHERE));
          bool numerical = mesh_sphere || !separation.allFinite() ||
              std::abs(separation.norm() - contact.distance) > 1e-5;
          if (numerical && robotPairWorldPoints(state, contact, world_points))
          {
            separation = world_points[0] - world_points[1];
            numerical = false;
          }
          // Retain the distance-only fallback for unsupported geometry;
          // normal robot mesh/sphere pairs now use exact known shape frames.
          if (numerical)
          {
            if (perturbed_distances.empty())
            {
              perturbed_distances.resize(n);
              auto probe_request = request;
              probe_request.distance_threshold += 0.001;
              for (int joint = 0; joint < n; ++joint)
              {
                auto probe = state;
                probe.setVariablePosition(arm_joint_names_[joint],
                    state.getVariablePosition(arm_joint_names_[joint]) + derivative_step);
                probe.update();
                if (self) scene_->getCollisionEnv()->distanceSelf(probe_request, perturbed_distances[joint], probe);
                else scene_->getCollisionEnv()->distanceRobot(probe_request, perturbed_distances[joint], probe);
              }
            }
            for (int joint = 0; joint < n; ++joint)
            {
              const auto found = perturbed_distances[joint].distances.find(pair.first);
              if (found == perturbed_distances[joint].distances.end() || found->second.empty())
              {
                reason = "COLLISION_DISTANCE_DERIVATIVE_UNAVAILABLE";
                return false;
              }
              gradient[joint] = (found->second.front().distance - contact.distance) / derivative_step;
            }
          }
          else
          {
            if (separation.norm() < 1e-9)
            {
              reason = "COLLISION_NORMAL_UNAVAILABLE";
              return false;
            }
            const Eigen::Vector3d normal = separation.normalized();
            for (int side = 0; side < 2; ++side)
            {
              if (contact.body_types[side] != collision_detection::BodyType::ROBOT_LINK) continue;
              const auto* link = model_->getLinkModel(contact.link_names[side]);
              if (!link || group->getUpdatedLinkModelsSet().count(link) == 0) continue;
              const Eigen::Vector3d point =
                  state.getGlobalLinkTransform(link).inverse() * world_points[side];
              Eigen::MatrixXd point_jacobian;
              if (!state.getJacobian(group, link, point, point_jacobian))
              {
                reason = "COLLISION_JACOBIAN_UNAVAILABLE";
                return false;
              }
              gradient += (side == 0 ? 1.0 : -1.0) * normal.transpose() * point_jacobian.topRows(3);
            }
          }
          if (!gradient.allFinite()) { reason = "NONFINITE_COLLISION_GRADIENT"; return false; }
          if (gradient.norm() < 1e-10) continue;  // This pair cannot change under arm motion.
          // A reference trajectory is not the measured robot. Reserve the
          // observed joint following error projected onto this pair's gap,
          // instead of treating a nominal 1 mm gap as exact while tracking.
          double tracking_margin = 0.0;
          double closing_speed = 0.0;
          double braking_time = 0.0;
          if (self)
            for (int joint = 0; joint < n; ++joint)
            {
              const auto reference = latest_arm_reference_.find(arm_joint_names_[joint]);
              if (reference != latest_arm_reference_.end())
                tracking_margin += std::abs(gradient[joint]) * std::abs(
                    reference->second - state.getVariablePosition(arm_joint_names_[joint]));
              const auto speed = latest_reference_velocity_.find(arm_joint_names_[joint]);
              const auto acceleration = reference_acceleration_.find(arm_joint_names_[joint]);
              if (speed != latest_reference_velocity_.end())
              {
                closing_speed -= gradient[joint] * speed->second;
                if (acceleration != reference_acceleration_.end() && acceleration->second > 0.0)
                  braking_time = std::max(braking_time,
                      std::abs(speed->second) / acceleration->second);
              }
            }
          // The measured 1 mm tangency was exhausted by tracking lag and
          // curved motion before the next command. Reserve the travel of the
          // already committed, acceleration-limited velocity while braking.
          // Normal close assembly pairs keep their established 1 mm margin;
          // moving non-adjacent arm/hand surfaces retain 5 mm plus this
          // measured-motion reserve. Binary self checks remain unchanged.
          bool structural_pair = false;
          if (self)
            for (const auto& pair : model_->getSRDF()->getDisabledCollisionPairs())
              if (pair.reason_ == "StructuralAdjacent" &&
                  ((pair.link1_ == contact.link_names[0] && pair.link2_ == contact.link_names[1]) ||
                   (pair.link2_ == contact.link_names[0] && pair.link1_ == contact.link_names[1])))
                structural_pair = true;
          const double clearance_margin = self && !structural_pair ? 0.005 : 0.001;
          const double braking_margin = self ?
              std::max(0.0, closing_speed) * (0.5 * braking_time + 0.02) : 0.0;
          rows.push_back(gradient);
          lower.push_back(-approach_rate * std::max(0.0,
              contact.distance - clearance_margin - tracking_margin - braking_margin));
        }
      }
    }
    if (have_robot_zero_)
    {
      // A bounded target alone did not bound the measured side angle: the
      // recorded arm passed -99 deg while its target remained at -88 deg.
      // Constrain the derivative of the ACTUAL C-relative angle inside the
      // same IK problem, including committed reference motion while braking.
      const Eigen::Vector3d tool_z = state.getGlobalLinkTransform("tool0").linear().col(2);
      const Eigen::Vector3d normal = robot_zero_rotation_.transpose() * tool_z;
      const double denominator = normal.y()*normal.y() + normal.z()*normal.z();
      Eigen::Matrix3d skew;
      skew << 0.0, -tool_z.z(), tool_z.y(),
              tool_z.z(), 0.0, -tool_z.x(),
              -tool_z.y(), tool_z.x(), 0.0;
      const Eigen::MatrixXd normal_jacobian =
          -robot_zero_rotation_.transpose() * skew * jacobian.bottomRows(3);
      if (denominator > 1e-6)
      {
        const double angle = std::atan2(-normal.y(), normal.z());
        const Eigen::RowVectorXd angle_gradient =
            (-normal.z()*normal_jacobian.row(1) + normal.y()*normal_jacobian.row(2)) / denominator;
        for (double sign : {-1.0, 1.0})
        {
          double following_margin = 0.0, outward_speed = 0.0, braking_time = 0.0;
          for (int joint = 0; joint < n; ++joint)
          {
            const auto reference = latest_arm_reference_.find(arm_joint_names_[joint]);
            if (reference != latest_arm_reference_.end())
              following_margin += std::abs(angle_gradient[joint]) * std::abs(
                  reference->second - state.getVariablePosition(arm_joint_names_[joint]));
            const auto rate = latest_reference_velocity_.find(arm_joint_names_[joint]);
            const auto acceleration = reference_acceleration_.find(arm_joint_names_[joint]);
            if (rate != latest_reference_velocity_.end())
            {
              outward_speed += sign * angle_gradient[joint] * rate->second;
              if (acceleration != reference_acceleration_.end() && acceleration->second > 0.0)
                braking_time = std::max(braking_time, std::abs(rate->second) / acceleration->second);
            }
          }
          const double reserve = following_margin +
              std::max(0.0, outward_speed) * (0.5*braking_time + 0.02);
          rows.push_back(-sign * angle_gradient);
          lower.push_back(-approach_rate * std::max(0.0,
              maximum_side_angle_ - sign*angle - 0.5*M_PI/180.0 - reserve));
        }
      }
      else if (maximum_side_angle_ <= M_PI/2.0 + 1e-8)
      {
        // At +/-90 deg tilt the Euler side angle is undefined. Its <=90
        // domain is still the closed hemisphere normal.z >= 0; do not divide
        // by the vanishing projection or invent a large angular correction.
        rows.push_back(normal_jacobian.row(2));
        lower.push_back(-approach_rate * std::max(0.0, normal.z()));
      }
    }
    const std::size_t linear_speed_rows = rows.size();
    const double requested_linear_speed = requested.head(3).norm();
    // A tiny position correction cannot authorize a fast sideways excursion
    // to achieve rotation. Directional progress alone admits arbitrarily
    // large perpendicular velocity, which accumulates drift at finite dt.
    // Retain transverse retreat freedom, bounded by the requested speed.
    for (int axis = 0; axis < 3; ++axis)
    {
      rows.push_back(jacobian.row(axis));
      lower.push_back(-requested_linear_speed);
    }
    const std::size_t collision_rows = rows.size();
    // Require progress toward each requested task, rather than fixing the
    // sign of every world-axis component. At the recorded folded wrist the
    // latter admits almost only zero: a coordinated retreat needs a small
    // transverse component before it can follow the direct Cartesian ray.
    // Position and rotation must each make nonnegative directional progress.
    // An unrequested task remains fixed (e.g. no rotation during translation).
    for (int block = 0; block < 2; ++block)
    {
      const Eigen::Vector3d direction = requested.segment<3>(3 * block);
      if (direction.norm() > 1e-9)
      {
        rows.push_back(direction.normalized().transpose() * jacobian.middleRows(3 * block, 3));
        lower.push_back(0.0);
        if (block == 0)
        {
          // Keep transverse corrections smaller than forward progress.
          // A purely tangential escape around a fixed position target is
          // not a user translation command; at finite dt it spirals away.
          const Eigen::Vector3d forward = direction.normalized();
          const Eigen::Vector3d across = forward.unitOrthogonal();
          const Eigen::Vector3d other = forward.cross(across);
          for (const auto& lateral : {across, other})
            for (double sign : {-1.0, 1.0})
            {
              rows.push_back((forward + sign * lateral).transpose() * jacobian.topRows(3));
              lower.push_back(0.0);
            }
        }
      }
      else
        for (int axis = 3 * block; axis < 3 * block + 3; ++axis)
        {
          rows.push_back(jacobian.row(axis));
          lower.push_back(0.0);
        }
    }
    constraint_count = rows.size();
    Eigen::MatrixXd weighted_jacobian = jacobian;
    Eigen::VectorXd target = requested;
    // Compare fractions of the existing linear/angular speed caps, not
    // metres with radians. The angular hierarchy below preserves the desired
    // rotation axis instead of trading it against position error.
    const double position_task_scale = 2.0 / 0.30;
    weighted_jacobian.topRows(3) *= position_task_scale;
    target.head(3) *= position_task_scale;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> hessian =
        weighted_jacobian.transpose() * weighted_jacobian;
    hessian.diagonal().array() += std::max(1e-6, inverse_damping_ * inverse_damping_);
    // Task normalization must include the weak-direction regularizer. The
    // position task was multiplied by 6.67 while these thresholds retained
    // their old scale: at the recorded nearly straight elbow the solver
    // repeatedly reversed joint_2/3, even for a constant Cartesian target.
    // Use the same scale here; strong directions retain the original cost.
    const double weak_direction_threshold = 0.06 * position_task_scale;
    const double weak_direction_damping = 0.08 * position_task_scale;
    Eigen::JacobiSVD<Eigen::MatrixXd> task_svd(weighted_jacobian, Eigen::ComputeFullV);
    for (Eigen::Index i = 0; i < task_svd.singularValues().size(); ++i)
    {
      const double ratio = task_svd.singularValues()[i] / weak_direction_threshold;
      const double damping_squared = weak_direction_damping * weak_direction_damping *
          std::max(0.0, 1.0 - ratio * ratio);
      hessian.noalias() += damping_squared * task_svd.matrixV().col(i) *
                          task_svd.matrixV().col(i).transpose();
    }
    Eigen::VectorXd gradient = -weighted_jacobian.transpose() * target;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> matrix(rows.size(), n);
    for (std::size_t i = 0; i < rows.size(); ++i) matrix.row(i) = rows[i];
    std::vector<double> lb(n), ub(n), upper(rows.size(), qpOASES::INFTY);
    for (int axis = 0; axis < 3; ++axis)
      upper[linear_speed_rows + axis] = requested_linear_speed;
    std::size_t task_row = collision_rows;
    for (int block = 0; block < 2; ++block)
    {
      if (requested.segment<3>(3 * block).norm() > 1e-9)
        task_row += block == 0 ? 5 : 1;
      else
        for (int axis = 0; axis < 3; ++axis)
          upper[task_row++] = 0.0;
    }
    for (int i = 0; i < n; ++i)
    {
      const auto& bound = model_->getVariableBounds(arm_joint_names_[i]);
      const double q = state.getVariablePosition(arm_joint_names_[i]);
      const double speed = bound.velocity_bounded_ ? bound.max_velocity_ : 1.0;
      lb[i] = -speed; ub[i] = speed;
      if (bound.position_bounded_)
      {
        lb[i] = std::max(lb[i], std::min(0.0, approach_rate *
                    (bound.min_position_ + joint_limit_margin_ - q)));
        ub[i] = std::min(ub[i], std::max(0.0, approach_rate *
                    (bound.max_position_ - joint_limit_margin_ - q)));
      }
    }
    qpOASES::Options options;
    options.setToReliable();
    // Zero satisfies every velocity interval. Starting all joints on their
    // lower speed bound made the recorded feasible QP cycle for 100 steps;
    // an inactive, zero-speed start solves that same QP without cycling.
    options.initialStatusBounds = qpOASES::ST_INACTIVE;
    options.printLevel = qpOASES::PL_NONE;
    // An unreachable position must not buy translation by tipping the palm.
    // A positive dot product with the requested angular velocity was not
    // enough: it allowed large perpendicular angular velocities. The Sep 10
    // recording reached 27 degrees of orientation error on an almost level
    // target. First find the feasible angular task under the SAME collision
    // and joint bounds; then optimize translation without changing that task.
    // Active operator rotation is retained, including combined 6-D motions.
    if (requested.tail(3).norm() > 1e-9)
    {
      // If the desired rotation cannot be executed, reduce its rate instead
      // of substituting a different rotation axis to go around a constraint.
      // The existing progress row selects the nonnegative direction.
      const Eigen::Vector3d angular_direction = requested.tail(3).normalized();
      const Eigen::Vector3d angular_normal = angular_direction.unitOrthogonal();
      const Eigen::Vector3d angular_binormal = angular_direction.cross(angular_normal);
      matrix.conservativeResize(rows.size() + 2, n);
      lower.resize(rows.size() + 2, 0.0);
      upper.resize(rows.size() + 2, 0.0);
      matrix.row(rows.size()) = angular_normal.transpose() * jacobian.bottomRows(3);
      matrix.row(rows.size() + 1) = angular_binormal.transpose() * jacobian.bottomRows(3);
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> angular_hessian =
          jacobian.bottomRows(3).transpose() * jacobian.bottomRows(3);
      // Both levels need the same weak-direction regularization. Otherwise
      // exact first-level angular tracking can demand opposing, near-limit
      // wrist velocities at a singularity, which level two cannot undo.
      angular_hessian += hessian - weighted_jacobian.transpose() * weighted_jacobian;
      const Eigen::VectorXd angular_gradient =
          -jacobian.bottomRows(3).transpose() * requested.tail(3);
      qpOASES::QProblem angular_problem(n, static_cast<int>(lower.size()), qpOASES::HST_POSDEF);
      angular_problem.setOptions(options);
      qpOASES::int_t angular_iterations = 100;
      std::vector<double> angular_solution(n, 0.0);
      const auto angular_result = angular_problem.init(
          angular_hessian.data(), angular_gradient.data(), matrix.data(),
          lb.data(), ub.data(), lower.data(), upper.data(), angular_iterations);
      if (angular_result != qpOASES::SUCCESSFUL_RETURN ||
          angular_problem.getPrimalSolution(angular_solution.data()) != qpOASES::SUCCESSFUL_RETURN)
      {
        reason = "CONSTRAINED_ANGULAR_SOLVE_FAILED:" + std::to_string(static_cast<int>(angular_result));
        return false;
      }
      const Eigen::Vector3d angular_task = jacobian.bottomRows(3) *
          Eigen::Map<Eigen::VectorXd>(angular_solution.data(), n);
      // Replace the single angular-progress row, avoiding redundant equality
      // constraints. Collision and positional-progress rows are unchanged.
      const std::size_t angular_row = collision_rows +
          (requested.head(3).norm() > 1e-9 ? 5 : 3);
      matrix.conservativeResize(angular_row + 3, n);
      lower.resize(angular_row + 3);
      upper.resize(angular_row + 3);
      for (int axis = 0; axis < 3; ++axis)
      {
        matrix.row(angular_row + axis) = jacobian.row(3 + axis);
        lower[angular_row + axis] = angular_task[axis];
        upper[angular_row + axis] = angular_task[axis];
      }
      constraint_count = lower.size();
    }
    qpOASES::QProblem problem(n, static_cast<int>(lower.size()), qpOASES::HST_POSDEF);
    problem.setOptions(options);
    qpOASES::int_t iterations = 100;
    const auto result = problem.init(hessian.data(), gradient.data(),
        rows.empty() ? nullptr : matrix.data(), lb.data(), ub.data(),
        rows.empty() ? nullptr : lower.data(), rows.empty() ? nullptr : upper.data(), iterations);
    velocity.assign(n, 0.0);
    if (result != qpOASES::SUCCESSFUL_RETURN ||
        problem.getPrimalSolution(velocity.data()) != qpOASES::SUCCESSFUL_RETURN)
    {
      reason = "CONSTRAINED_VELOCITY_SOLVE_FAILED:" + std::to_string(static_cast<int>(result));
      return false;
    }
    Eigen::Map<Eigen::VectorXd> solved(velocity.data(), n);
    if (!solved.allFinite()) { reason = "NONFINITE_PROJECTED_VELOCITY"; return false; }
    const Eigen::VectorXd twist = jacobian * solved;
    const double scale = std::max({1.0, twist.head(3).norm() / 0.30, twist.tail(3).norm() / 2.0});
    solved /= scale;  // Zero is feasible, so uniform reduction preserves all inequalities.
    reason = "SAFE";
    return true;
  }

  std::map<std::string, double> fingerObjectClearances(moveit::core::RobotState& state) const
  {
    // Release is complete only after actual finger surfaces have separated,
    // not merely because the OPEN joint command has finished. Query these
    // pairs independently of the temporarily allowed grasp contacts.
    const std::vector<std::string> fingers{
        "f1link2", "f1link3", "f2link1", "f2link2", "f3link2", "f3link3"};
    const std::set<std::string> targets{
        "target_object", "ground_object_left", "ground_object_right"};
    std::set<const moveit::core::LinkModel*> active;
    collision_detection::AllowedCollisionMatrix acm(scene_->getAllowedCollisionMatrix());
    std::map<std::string, double> result;
    for (const auto& name : scene_->getWorld()->getObjectIds())
    {
      if (targets.count(name)) result[name] = 0.02;  // No pair within 20 mm means clear.
      for (const auto& finger : fingers) acm.setEntry(finger, name, targets.count(name) == 0);
    }
    for (const auto& name : fingers)
      if (model_->hasLinkModel(name)) active.insert(model_->getLinkModel(name));
    collision_detection::DistanceRequest request;
    request.active_components_only = &active;
    request.acm = &acm;
    request.enable_signed_distance = true;
    request.type = collision_detection::DistanceRequestType::SINGLE;
    request.distance_threshold = 0.02;
    collision_detection::DistanceResult distances;
    state.updateCollisionBodyTransforms();
    scene_->getCollisionEnv()->distanceRobot(request, distances, state);
    for (const auto& pair : distances.distances)
      for (const auto& entry : pair.second)
        for (const auto& name : entry.link_names)
          if (targets.count(name) && std::isfinite(entry.distance))
            result[name] = std::min(result[name], entry.distance);
    return result;
  }

  // Check the actual shaped command immediately before MoveIt Servo. A
  // shaper after this gate could reintroduce an obsolete blocked approach.
  void filterCartesianCommand(const handarm_moveit_demo::HandCommandConstPtr& message)
  {
    const auto began = ros::WallTime::now();
    auto output = *message;
    Eigen::Matrix<double, 6, 1> requested;
    requested << message->twist.linear.x, message->twist.linear.y,
        message->twist.linear.z, message->twist.angular.x,
        message->twist.angular.y, message->twist.angular.z;
    Eigen::Matrix<double, 6, 1> selected = Eigen::Matrix<double, 6, 1>::Zero();
    Eigen::Matrix<double, 6, 1> measured = Eigen::Matrix<double, 6, 1>::Zero();
    std::string reason = "INPUT_INVALID";
    std::vector<double> projected_joint_velocity;
    bool projected_command_safe = false;
    double projected_deadline = 0.0;
    std::size_t projection_constraint_count = 0;
    std::map<std::string, double> finger_object_clearance;
    Clearance current{ 0.0, 0.0 };
    double condition = 0.0;
    bool singularity_limited = false;
    bool joint_limit_limited = false;
    bool reset_singularity_allowed = false;
    bool reset_joint_limit_allowed = false;
    std::array<bool, 3> recovery_limited_axes{{false, false, false}};
    bool recovery_rotation_limited = false;
    double retreat_gain = 1.0;
    moveit::core::RobotState state(model_);
    std::map<std::string, double> hand_reference;
    std::map<std::string, double> arm_reference;
    bool ready;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ready = have_complete_state_ && have_scene_snapshot_ && !fault_latched_ &&
          !hand_motion_active_ &&
          (began - latest_state_wall_time_).toSec() <= state_timeout_s_;
      state = *latest_state_;
      hand_reference = latest_hand_reference_;
      arm_reference = latest_arm_reference_;
      projected_deadline = latest_state_wall_time_.toSec() + prediction_horizon_s_;
    }
    const double age = (ros::Time::now() - message->header.stamp).toSec();
    // /clock is delivered independently to each process. Allow at most one
    // 50 Hz command period of simulation clock skew, without re-stamping old
    // commands or extending their 150 ms stale-input limit.
    const double future_tolerance = ros::Time::isSimTime() ? 0.02 : 0.0;
    if (!ready)
    {
      output.valid = false;
      reason = "WAITING_FOR_FRESH_ROBOT_AND_SCENE";
    }
    else if (message->valid && requested.allFinite() && age >= -future_tolerance && age <= 0.15 &&
             message->header.frame_id == "base_link")
    {
      current = clearance(state);
      if (constrained_velocity_projection_)
        finger_object_clearance = fingerObjectClearances(state);
      // This guard sees the full compliant hand. It must recognize its own
      // contact state even when Servo's independently updated scene/hand
      // snapshot reports scale 1. Permit a checked, non-worsening partial
      // retreat; do not require clearing the entire contact in one probe.
      if (current.self < 0.0 || current.world < 0.0)
        proximity_recovery_active_ = true;
      Clearance selected_clearance = current;
      const auto* group = model_->getJointModelGroup("abbarm");
      Eigen::MatrixXd jacobian;
      state.getJacobian(group, model_->getLinkModel("tool0"),
                        Eigen::Vector3d::Zero(), jacobian);
      std::vector<double> joint_velocity;
      state.copyJointGroupVelocities(group, joint_velocity);
      measured = jacobian * Eigen::Map<Eigen::VectorXd>(
          joint_velocity.data(), joint_velocity.size());
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(
          jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
      const auto singular = svd.singularValues();
      Eigen::VectorXd inverse_values = singular;
      for (Eigen::Index i = 0; i < inverse_values.size(); ++i)
        inverse_values[i] = inverse_damping_ > 0.0 ?
            singular[i] / (singular[i] * singular[i] + inverse_damping_ * inverse_damping_) :
            (singular[i] > 1e-12 ? 1.0 / singular[i] : 0.0);
      const Eigen::MatrixXd inverse = svd.matrixV() * inverse_values.asDiagonal() * svd.matrixU().transpose();
      condition = singular[0] / std::max(1.0e-12, singular[singular.size() - 1]);
      const double current_margin = armMargin(state);
      double selected_condition = condition, selected_margin = current_margin;
      joint_limit_limited = current_margin <= joint_limit_margin_ + 0.01;
      // Status 1 can persist after leaving a singularity. Use current
      // geometry, and flag only the strongly attenuated part of Servo's
      // existing range; do not lower its approach/stop thresholds.
      singularity_limited = inverse_damping_ <= 0.0 && condition >= singularity_hard_ - proximity_enter_scale_ *
          (singularity_hard_ - singularity_soft_);
      std::vector<double> position;
      state.copyJointGroupPositions(group, position);
      bool swept_rejected = false;
      const auto admissible = [&](const Eigen::Matrix<double, 6, 1>& velocity,
                                  Clearance* predicted_clearance = nullptr,
                                  double* predicted_condition = nullptr,
                                  double* predicted_margin = nullptr) {
        swept_rejected = false;
        Eigen::VectorXd step = inverse * velocity * 0.06;
        if (!step.allFinite()) return false;
        // A small directional probe; downstream swept checks own the full
        // braking horizon. Keep it local even near ill-conditioned poses.
        const double largest = step.cwiseAbs().maxCoeff();
        if (largest > 0.01) step *= 0.01 / largest;
        auto candidate = state;
        auto predicted = position;
        for (std::size_t i = 0; i < predicted.size(); ++i) predicted[i] += step[i];
        candidate.setJointGroupPositions(group, predicted);
        preserveMeasuredHandDeflection(state, candidate);
        candidate.update();
        // This probe changes arm joints only. The unchanged compliant hand
        // has already passed the measured-state bound tolerance above. In
        // the side-grasp replay f2j1=-0.00017 rad (normal contact deflection)
        // made an all-joint, zero-tolerance check reject every arm direction.
        // Keep predicted arm limits exact without reclassifying that valid
        // hand measurement as a new out-of-bounds arm command.
        if (!candidate.satisfiesBounds(group, 0.0)) return false;
        const double next_margin = armMargin(candidate);
        Eigen::MatrixXd next_jacobian;
        candidate.getJacobian(group, model_->getLinkModel("tool0"),
                              Eigen::Vector3d::Zero(), next_jacobian);
        Eigen::JacobiSVD<Eigen::MatrixXd> next_svd(next_jacobian);
        const auto next_singular = next_svd.singularValues();
        const double next_condition = next_singular[0] /
            std::max(1e-12, next_singular[next_singular.size() - 1]);
        if (predicted_condition) *predicted_condition = next_condition;
        if (predicted_margin) *predicted_margin = next_margin;
        if (next_margin < std::min(current_margin, joint_limit_margin_) - 1e-6)
          return false;
        if (inverse_damping_ <= 0.0 && condition >= singularity_hard_ && next_condition > condition + 1e-6)
          return false;
        const Clearance next = clearance(candidate);
        if (predicted_clearance) *predicted_clearance = next;
        // Use the downstream arm/hand collision test here as well, before
        // selecting feasible Cartesian components. Otherwise this layer can
        // approve all axes while the final gate rejects their combination.
        if (enable_swept_command_gate_)
        {
          Eigen::VectorXd joint_rates = inverse * velocity;
          double utilization = 1.0;
          for (std::size_t i = 0; i < arm_joint_names_.size(); ++i)
          {
            const auto& bounds = model_->getVariableBounds(arm_joint_names_[i]);
            if (bounds.velocity_bounded_)
              utilization = std::max(utilization,
                  std::abs(joint_rates[i]) / bounds.max_velocity_);
          }
          joint_rates /= utilization;
          std::vector<double> rates(joint_rates.data(),
                                    joint_rates.data() + joint_rates.size());
          std::string swept_reason;
          if (!predictedVelocityIsSafe(rates, state, hand_reference, arm_reference, swept_reason))
          {
            swept_rejected = true;
            return false;
          }
        }
        if (!proximity_recovery_active_)
          return next.self >= 0.0 && next.world >= 0.0;
        // Preserve room for a retreat once Servo is strongly attenuating.
        // Allowing a still-clear approach here was directly tested: at the
        // elevated side pose, extra rotation drove scale below 0.01 and made
        // subsequent side/down translation almost stationary. Keep feasible
        // retreat components available instead of spending that clearance.
        return next.self >= std::min(current.self, 0.01) - 1.0e-6 &&
               next.world >= std::min(current.world, 0.03) - 1.0e-6;
      };
      // A direction can be safe while the requested 80 ms travel is too long.
      // Reduce only such swept-path failures, preserving the requested twist
      // direction. Never turn a worsening local collision/limit direction into
      // an accepted command by merely making its probe smaller.
      const auto admissible_at_safe_speed = [&](Eigen::Matrix<double, 6, 1>& velocity,
                                                Clearance* next,
                                                double* next_condition,
                                                double* next_margin) {
        const Eigen::VectorXd rates = inverse * velocity;
        double utilization = 1.0;
        for (std::size_t i = 0; i < arm_joint_names_.size(); ++i)
        {
          const auto& bounds = model_->getVariableBounds(arm_joint_names_[i]);
          if (bounds.velocity_bounded_)
            utilization = std::max(utilization, std::abs(rates[i]) / bounds.max_velocity_);
        }
        velocity /= utilization;
        for (int attempt = 0; attempt < 4; ++attempt)
        {
          if (admissible(velocity, next, next_condition, next_margin)) return true;
          if (!swept_rejected) return false;
          velocity *= 0.5;
        }
        return false;
      };
      if (constrained_velocity_projection_)
      {
        projected_command_safe = projectJointVelocity(state, jacobian, requested,
            projected_joint_velocity, projection_constraint_count, reason);
        if (projected_command_safe)
          projected_command_safe = predictedVelocityIsSafe(projected_joint_velocity,
              state, hand_reference, arm_reference, reason);
        bool translation_retreat = false;
        if (!projected_command_safe && requested.head<3>().norm() > 1e-9 &&
            (reason.find("PREDICTED_WORLD_COLLISION:") == 0 ||
             reason.find("CONTACT_NOT_SEPARATING:") == 0))
        {
          // At the recorded table contact, the combined rotation/translation
          // QP was locally tangent, but its acceleration-limited trajectory
          // pressed the fingertip farther in. Translation alone separated
          // safely. Rejecting that one combined solution must not discard
          // the operator's feasible exit. Preserve the original pose target;
          // hold angular velocity only for this checked execution step.
          auto linear = requested;
          linear.tail<3>().setZero();
          std::vector<double> retreat_velocity;
          std::size_t retreat_constraints = 0;
          std::string retreat_reason;
          if (projectJointVelocity(state, jacobian, linear, retreat_velocity,
                  retreat_constraints, retreat_reason) &&
              predictedVelocityIsSafe(retreat_velocity, state, hand_reference,
                  arm_reference, retreat_reason))
          {
            projected_joint_velocity = std::move(retreat_velocity);
            projection_constraint_count = retreat_constraints;
            projected_command_safe = true;
            translation_retreat = true;
          }
        }
        if (projected_command_safe)
        {
          selected = jacobian * Eigen::Map<Eigen::VectorXd>(projected_joint_velocity.data(),
                                                         projected_joint_velocity.size());
          reason = translation_retreat ? "TRANSLATION_ALLOWED_ROTATION_WOULD_PRESS_WORLD" :
              (selected - requested).norm() < 1e-5 ?
              "FULL_COMMAND_ALLOWED" : "FEASIBLE_TRANSLATION_AND_ROTATION";
        }
      }
      else
      {
      auto full = requested;
      if (requested.norm() <= 1.0e-12 || admissible_at_safe_speed(full, &selected_clearance,
                                                   &selected_condition, &selected_margin))
      {
        selected = full;
        reason = (full - requested).norm() > 1e-9 ? "DIRECTION_ALLOWED_SPEED_REDUCED" : "FULL_COMMAND_ALLOWED";
      }
      else
      {
        Eigen::Matrix<double, 6, 1> linear = requested, angular = requested;
        linear.tail<3>().setZero();
        angular.head<3>().setZero();
        if (linear.norm() > 1.0e-12 && admissible_at_safe_speed(linear, &selected_clearance,
                                                &selected_condition, &selected_margin))
        {
          selected = linear;
          reason = "TRANSLATION_ALLOWED_ROTATION_BLOCKED";
        }
        else
        {
          // One approach axis must not suppress a different, feasible escape
          // axis. Retain requested Cartesian components only, and recheck the
          // complete retained combination after each addition.
          std::vector<int> axes{ 0, 1, 2 };
          std::stable_sort(axes.begin(), axes.end(), [&](int a, int b) {
            return std::abs(requested[a]) > std::abs(requested[b]);
          });
          for (int axis : axes)
          {
            if (std::abs(requested[axis]) <= 1.0e-12) continue;
            auto candidate = selected;
            candidate[axis] = requested[axis];
            Clearance next;
            double next_condition, next_margin;
            if (admissible_at_safe_speed(candidate, &next, &next_condition, &next_margin))
            {
              selected = candidate;
              selected_clearance = next;
              selected_condition = next_condition;
              selected_margin = next_margin;
            }
          }
          reason = selected.norm() > 1e-12 ? "FEASIBLE_TRANSLATION_COMPONENTS" : "APPROACH_BLOCKED";
        }
        Clearance next;
        double next_condition, next_margin;
        Eigen::Matrix<double, 6, 1> combined = selected + angular;
        if (angular.norm() > 1.0e-12 && admissible_at_safe_speed(combined, &next,
                                                   &next_condition, &next_margin))
        {
          selected = combined;
          selected_clearance = next;
          selected_condition = next_condition;
          selected_margin = next_margin;
          reason = "FEASIBLE_TRANSLATION_AND_ROTATION";
        }
      }
      }
      if (selected.norm() > 1e-9)
      {
        reset_singularity_allowed = selected_condition < condition - 1e-5 ||
            (!singularity_limited && selected_condition < singularity_hard_);
        reset_joint_limit_allowed = selected_margin > current_margin + 1e-6 ||
            selected_margin > joint_limit_margin_;
      }
      // Near a joint/singularity limit, do not shrink all six target axes.
      // Attribute an unexecuted excursion only to a requested component
      // which moves farther into the active constraint. A simultaneous
      // lowering/retreat component must keep its full hand mapping.
      const bool approaching_singularity = singularity_limited &&
          selected_condition > condition + 1e-6;
      const bool approaching_joint_limit = joint_limit_limited &&
          selected_margin < current_margin - 1e-6;
      for (int axis = 0; axis < 4; ++axis)
      {
        Eigen::Matrix<double, 6, 1> component = Eigen::Matrix<double, 6, 1>::Zero();
        if (axis < 3) component[axis] = requested[axis];
        else component.tail<3>() = requested.tail<3>();
        // Slowing a still-executing direction must not collapse its C mapping
        // as if it were blocked. Recovery removes accumulated targets only for
        // directions actually rejected here or worsening an active hard limit.
        bool limited = axis < 3 ?
            (std::abs(requested[axis]) > 1e-9 && selected[axis] * requested[axis] <= 1e-12) :
            (requested.tail<3>().norm() > 1e-9 && selected.tail<3>().norm() <= 1e-9);
        if (!limited && component.norm() > 1e-9 &&
            (approaching_singularity || approaching_joint_limit))
        {
          double next_condition = condition, next_margin = current_margin;
          admissible(component, nullptr, &next_condition, &next_margin);
          limited = (approaching_singularity && next_condition > condition + 1e-6) ||
                    (approaching_joint_limit && next_margin < current_margin - 1e-6);
        }
        if (axis < 3) recovery_limited_axes[axis] = limited;
        else recovery_rotation_limited = limited;
      }
      const bool exiting_proximity =
          (current.self < 0.01 && selected_clearance.self > current.self + 1.0e-6) ||
          (current.world < 0.03 && selected_clearance.world > current.world + 1.0e-6);
      if (proximity_recovery_active_ && exiting_proximity && selected.norm() > 1e-12)
      {
        // Suggest compensation upstream of acceleration shaping. The next
        // shaped command is independently rechecked here before publication.
        // Servo checks the actual retreat direction before applying its
        // proximity scale. Multiplying the upstream target by five stored
        // an excessive request that could burst through when the band ended.
        retreat_gain = 1.0;
        reason += "_CLEARANCE_INCREASING_RETREAT";
      }
    }
    else output.valid = false;
    if (constrained_velocity_projection_)
    {
      // In this opt-in mode the QP is the sole joint-velocity producer. Servo
      // continues collision monitoring, but its direction-blind velocity
      // scaling must not undo an already constrained escape command.
      last_raw_command_wall_time_ = began;
      have_raw_command_ = true;
      if (projected_command_safe && ros::WallTime::now().toSec() < projected_deadline)
      {
        std_msgs::Float64MultiArray safe;
        safe.data = projected_joint_velocity;
        safe.data.push_back(projected_deadline);
        safe_velocity_publisher_.publish(safe);
        publishCommandStatus(true, "SAFE_CONSTRAINED_VELOCITY");
      }
      else
      {
        selected.setZero();
        if (projected_command_safe) reason = "PREDICTION_EXPIRED_BEFORE_PUBLICATION";
        publishZeroVelocity();
        publishCommandStatus(false, reason);
      }
    }
    output.twist.linear.x = selected[0]; output.twist.linear.y = selected[1];
    output.twist.linear.z = selected[2]; output.twist.angular.x = selected[3];
    output.twist.angular.y = selected[4]; output.twist.angular.z = selected[5];
    geometric_command_publisher_.publish(output);
    std::ostringstream stream;
    const bool translation_limited = output.valid &&
        (selected.head<3>() - requested.head<3>()).norm() > 1e-9;
    const bool rotation_limited = output.valid &&
        (selected.tail<3>() - requested.tail<3>()).norm() > 1e-9;
    stream << "{\"reason\":\"" << reason << "\",\"self_distance_m\":" << current.self
           << ",\"constrained_velocity_projection\":" << (constrained_velocity_projection_ ? "true" : "false")
           << ",\"side_limit_reference_ready\":" << (have_robot_zero_ ? "true" : "false")
           << ",\"maximum_side_angle_deg\":" << maximum_side_angle_ * 180.0/M_PI
           << ",\"projection_constraints\":" << projection_constraint_count
           << ",\"input_age_s\":" << age
           << ",\"input_valid\":" << (message->valid ? "true" : "false")
           << ",\"world_distance_m\":" << current.world << ",\"processing_ms\":"
           << (ros::WallTime::now() - began).toSec() * 1000.0
           << ",\"filtered\":" << ((selected - requested).norm() > 1e-9 ? "true" : "false")
           << ",\"translation_limited\":"
           << (translation_limited ? "true" : "false")
           << ",\"rotation_limited\":"
           << (rotation_limited ? "true" : "false")
           << ",\"jacobian_condition\":" << condition
           << ",\"singularity_limited\":" << (singularity_limited ? "true" : "false")
           << ",\"joint_limit_limited\":" << (joint_limit_limited ? "true" : "false")
           << ",\"retreat_gain\":" << retreat_gain
           << ",\"reset_singularity_allowed\":" << (reset_singularity_allowed ? "true" : "false")
           << ",\"reset_joint_limit_allowed\":" << (reset_joint_limit_allowed ? "true" : "false")
           << ",\"recovery_rotation_limited\":" << (recovery_rotation_limited ? "true" : "false")
           << ",\"recovery_limited_axes\":[";
    for (int i = 0; i < 3; ++i)
      stream << (i ? "," : "") << (recovery_limited_axes[i] ? "true" : "false");
    stream << "]"
           << ",\"translation_limited_axes\":[";
    for (int i = 0; i < 3; ++i)
      stream << (i ? "," : "") << (output.valid && std::abs(selected[i] - requested[i]) > 1e-9 ? "true" : "false");
    stream << "],\"approved_velocity\":[";
    for (int i = 0; i < 6; ++i) stream << (i ? "," : "") << selected[i];
    stream << "],\"measured_velocity\":[";
    for (int i = 0; i < 6; ++i) stream << (i ? "," : "") << measured[i];
    stream << "],\"finger_object_clearance_m\":{";
    bool first_clearance = true;
    for (const auto& item : finger_object_clearance)
    {
      stream << (first_clearance ? "" : ",") << "\"" << item.first << "\":" << item.second;
      first_clearance = false;
    }
    stream << "}}";
    std_msgs::String diagnostic; diagnostic.data = stream.str();
    geometric_diagnostic_publisher_.publish(diagnostic);
  }

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

  void validateHandReference()
  {
    if (hand_reference_joint_names_.empty() ||
        hand_reference_joint_names_.size() != initial_hand_reference_.size())
      throw std::runtime_error(
          "hand reference names and initial positions must be non-empty and equal-sized");
    std::set<std::string> unique;
    for (std::size_t index = 0;
         index < hand_reference_joint_names_.size(); ++index)
    {
      const std::string& name = hand_reference_joint_names_[index];
      const double value = initial_hand_reference_[index];
      if (!unique.insert(name).second)
        throw std::runtime_error("hand reference contains a duplicate: " + name);
      if (std::find(required_joint_names_.begin(), required_joint_names_.end(),
                    name) == required_joint_names_.end())
        throw std::runtime_error(
            "hand reference joint is not a required joint: " + name);
      if (std::find(arm_joint_names_.begin(), arm_joint_names_.end(), name) !=
          arm_joint_names_.end())
        throw std::runtime_error("arm joint cannot be a hand reference: " + name);
      const moveit::core::JointModel* joint = model_->getJointOfVariable(name);
      if (!joint || joint->getMimic() || !std::isfinite(value) ||
          !joint->satisfiesPositionBounds(&value, 0.0))
        throw std::runtime_error(
            "invalid initial hand reference for joint: " + name);
      latest_hand_reference_[name] = value;
    }
  }

  bool positionsSatisfyBounds(const moveit::core::RobotState& state,
                              double bounds_tolerance_rad,
                              std::string* error,
                              const moveit::core::RobotState* accepted_hand_start = nullptr) const
  {
    bool valid = true;
    std::ostringstream details;
    details << "joint limit violation";
    for (const moveit::core::JointModel* joint : model_->getActiveJointModels())
    {
      if (joint->satisfiesPositionBounds(
              state.getVariablePositions() + joint->getFirstVariableIndex(),
              bounds_tolerance_rad))
        continue;
      // A compliant measured finger may start outside its nominal stop.
      // Arm motion is allowed to relieve that contact; the predicted finger
      // must not create a new violation or deepen the existing one.
      if (accepted_hand_start && joint->getVariableCount() == 1)
      {
        const auto& name = joint->getVariableNames().front();
        if (std::find(hand_reference_joint_names_.begin(), hand_reference_joint_names_.end(), name) !=
            hand_reference_joint_names_.end())
        {
          const auto& bound = model_->getVariableBounds(name);
          const double before = accepted_hand_start->getVariablePosition(name);
          const double after = state.getVariablePosition(name);
          if (after >= std::min(bound.min_position_, before) - 1e-9 &&
              after <= std::max(bound.max_position_, before) + 1e-9) continue;
        }
      }
      valid = false;
      const std::vector<std::string>& names = joint->getVariableNames();
      const std::vector<moveit::core::VariableBounds>& bounds =
          joint->getVariableBounds();
      for (std::size_t index = 0; index < names.size(); ++index)
      {
        const double value = state.getVariablePosition(names[index]);
        if (bounds[index].position_bounded_ &&
            (value < bounds[index].min_position_ - bounds_tolerance_rad ||
             value > bounds[index].max_position_ + bounds_tolerance_rad))
        {
          details << ":" << names[index] << "=" << value
                  << " not_in[" << bounds[index].min_position_
                  << "," << bounds[index].max_position_ << "]";
        }
      }
    }
    if (!valid && error)
      *error = details.str();
    return valid;
  }

  bool onlyHandPositionLimitsViolated(const moveit::core::RobotState& state) const
  {
    bool hand_violation = false;
    for (const auto* joint : model_->getActiveJointModels())
    {
      if (joint->satisfiesPositionBounds(
              state.getVariablePositions() + joint->getFirstVariableIndex(),
              measured_joint_limit_tolerance_rad_))
        continue;
      for (const auto& name : joint->getVariableNames())
        if (std::find(hand_reference_joint_names_.begin(),
                      hand_reference_joint_names_.end(), name) ==
            hand_reference_joint_names_.end())
          return false;
      hand_violation = true;
    }
    return hand_violation;
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
    // RobotState::satisfiesBounds() also checks stored velocity bounds.  The
    // state retains measured velocities for coast prediction, so using the
    // generic API here can reject a perfectly legal measured position merely
    // because a noisy velocity estimate exceeds the URDF velocity limit.
    if (!positionsSatisfyBounds(state, bounds_tolerance_rad, &error))
      return false;
    return true;
  }

  bool collisionFree(moveit::core::RobotState& state,
                     collision_detection::CollisionResult& result) const
  {
    collision_detection::CollisionRequest request;
    request.contacts = true;
    request.max_contacts = 1;
    request.max_contacts_per_pair = 1;
    scene_->checkSelfCollision(request, result, state, *strict_acm_);
    return !result.collision;
  }

  std::string stateText(const moveit::core::RobotState& state) const
  {
    std::ostringstream stream;
    stream << "q=[";
    for (std::size_t index = 0; index < required_joint_names_.size(); ++index)
    {
      if (index)
        stream << ",";
      stream << required_joint_names_[index] << "="
             << state.getVariablePosition(required_joint_names_[index]);
    }
    stream << "]:qd=[";
    for (std::size_t index = 0; index < required_joint_names_.size(); ++index)
    {
      if (index)
        stream << ",";
      stream << required_joint_names_[index] << "="
             << (state.hasVelocities() ?
                    state.getVariableVelocity(required_joint_names_[index]) :
                    0.0);
    }
    stream << "]";
    return stream.str();
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
    zero.data.push_back(0.0);  // private safe-command contract: expired now
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
                               const std::map<std::string, double>& hand_reference,
                               const std::map<std::string, double>& arm_reference,
                               std::string& reason) const
  {
    // An incidental fingertip contact must not make every retreat impossible.
    // Compare contact depth along the actual bounded reference prediction;
    // merely remaining in contact for the first part of a retreat is not a
    // reason to reject it. Never permit a new pair or a deeper existing pair.
    std::map<std::pair<std::string, std::string>, double> starting_contacts;
    if (constrained_velocity_projection_)
    {
      const auto contacts = worldContacts(start);
      if (contacts.contact_count >= 200 || (contacts.collision && contacts.contacts.empty()))
      {
        reason = "WORLD_CONTACT_DETAILS_UNAVAILABLE";
        return false;
      }
      for (const auto& pair : contacts.contacts)
      {
        if (!robotWorldContact(pair.first.first, pair.first.second))
        {
          reason = "CURRENT_CONTACT:" + pair.first.first + "," + pair.first.second;
          return false;
        }
        for (const auto& contact : pair.second)
          starting_contacts[pair.first] = std::max(starting_contacts[pair.first], contact.depth);
      }
    }
    // Match the EGM reference integrator. A stopped actuator with an 8 rad/s²
    // acceleration limit can advance only 0.0256 rad in 80 ms, not the 0.35
    // rad implied by instantly applying a new 4.36 rad/s request. Include the
    // existing position lead and both acceleration and reversal/braking.
    double coordinated_ramp_s = 0.0;
    if (constrained_velocity_projection_)
      for (std::size_t i = 0; i < arm_joint_names_.size(); ++i)
      {
        const auto initial = latest_reference_velocity_.find(arm_joint_names_[i]);
        const auto acceleration = reference_acceleration_.find(arm_joint_names_[i]);
        if (initial != latest_reference_velocity_.end() && acceleration != reference_acceleration_.end())
          coordinated_ramp_s = std::max(coordinated_ramp_s,
              std::abs(velocity[i] - initial->second) / acceleration->second);
      }
    const auto reference_travel = [this, coordinated_ramp_s](const std::string& name, double target, double t) {
      const auto initial = latest_reference_velocity_.find(name);
      const auto acceleration = reference_acceleration_.find(name);
      if (initial == latest_reference_velocity_.end() || acceleration == reference_acceleration_.end())
        return target * t;
      const double v0 = initial->second;
      if (constrained_velocity_projection_ && coordinated_ramp_s > 1e-12)
      {
        const double ramp = std::min(t, coordinated_ramp_s);
        return v0 * ramp + 0.5 * (target - v0) * ramp * ramp / coordinated_ramp_s +
               target * (t - ramp);
      }
      const double a = std::copysign(acceleration->second, target - v0);
      const double ramp = std::min(t, std::abs(target - v0) / acceleration->second);
      return v0 * ramp + 0.5 * a * ramp * ramp + target * (t - ramp);
    };
    const auto check_scenario = [this, &start, &reason, &arm_reference, &reference_travel,
                                &starting_contacts](
        const std::map<std::string, double>& predicted_velocity,
        const std::string& source) -> bool
    {
      double maximum_travel = 0.0;
      for (const auto& item : predicted_velocity)
      {
        const auto reference = arm_reference.find(item.first);
        const double lead = reference == arm_reference.end() ? 0.0 :
            reference->second - start.getVariablePosition(item.first);
        double speed_bound = std::abs(item.second);
        const auto initial = latest_reference_velocity_.find(item.first);
        const auto acceleration = reference_acceleration_.find(item.first);
        if (initial != latest_reference_velocity_.end() && acceleration != reference_acceleration_.end())
        {
          const double v0 = initial->second;
          const double end = v0 + std::max(-acceleration->second * prediction_horizon_s_,
              std::min(acceleration->second * prediction_horizon_s_, item.second - v0));
          speed_bound = std::max(std::abs(v0), std::abs(end));
        }
        maximum_travel = std::max(maximum_travel,
            std::abs(lead) + speed_bound * prediction_horizon_s_);
      }
      const std::size_t samples = std::max<std::size_t>(
              1, static_cast<std::size_t>(
                     std::ceil(maximum_travel /
                               maximum_prediction_step_rad_)));
      if (samples > static_cast<std::size_t>(maximum_prediction_samples_))
      {
        reason = "PREDICTION_RESOLUTION_LIMIT";
        return false;
      }
      moveit::core::RobotState candidate(start);
      auto previous_contacts = starting_contacts;
      for (std::size_t sample = 1; sample <= samples; ++sample)
      {
        const double prediction_time = prediction_horizon_s_ *
            static_cast<double>(sample) / static_cast<double>(samples);
        candidate = start;
        for (const auto& item : predicted_velocity)
        {
          const auto reference = arm_reference.find(item.first);
          const double lead = reference == arm_reference.end() ? 0.0 :
              reference->second - start.getVariablePosition(item.first);
          candidate.setVariablePosition(
              item.first,
              start.getVariablePosition(item.first) +
                  lead * prediction_time / prediction_horizon_s_ +
                  reference_travel(item.first, item.second, prediction_time));
        }
        preserveMeasuredHandDeflection(start, candidate);
        candidate.update();
        std::string bounds_error;
        if (!positionsSatisfyBounds(candidate, 0.0, &bounds_error,
            constrained_velocity_projection_ && recover_simulated_hand_limit_transients_ &&
                ros::Time::isSimTime() ? &start : nullptr))
        {
          std::ostringstream stream;
          stream << "PREDICTED_JOINT_LIMIT:" << bounds_error
                 << ":sample=" << sample << "/"
                 << samples << ":source=" << source;
          reason = stream.str();
          return false;
        }
        collision_detection::CollisionResult result;
        if (!collisionFree(candidate, result))
        {
          std::ostringstream stream;
          stream << "PREDICTED_SELF_COLLISION:" << pairText(result)
                 << ":sample=" << sample << "/" << samples
                 << ":source=" << source;
          reason = stream.str();
          return false;
        }
        if (constrained_velocity_projection_)
        {
          const auto world_result = worldContacts(candidate);
          if (world_result.contact_count >= 200 ||
              (world_result.collision && world_result.contacts.empty()))
          {
            reason = "WORLD_CONTACT_DETAILS_UNAVAILABLE";
            return false;
          }
          // A finger already stopped by the object cannot continue through
          // it at its unloaded motor-reference speed. In the recorded pose
          // that extrapolation invented another 0.44 mm of penetration and
          // vetoed an upward arm retreat. Keep the complete commanded-hand
          // check for every NEW pair; assess escape from an existing finger
          // contact using the measured compliant hand carried by the arm.
          for (const auto& pair : world_result.contacts)
          {
            if (starting_contacts.count(pair.first) == 0)
            {
              reason = "PREDICTED_WORLD_COLLISION:" + pair.first.first + "," + pair.first.second;
              return false;
            }
          }
          auto retreat_contacts = world_result;
          if (!starting_contacts.empty())
          {
            auto loaded_hand = candidate;
            for (const auto& name : model_->getVariableNames())
              if (std::find(arm_joint_names_.begin(), arm_joint_names_.end(), name) == arm_joint_names_.end())
                loaded_hand.setVariablePosition(name, start.getVariablePosition(name));
            // Restore passive deflection after independent variables, whose
            // updates can propagate the model's unloaded mimic relation.
            for (const auto* joint : model_->getMimicJointModels())
              for (const auto& name : joint->getVariableNames())
                loaded_hand.setVariablePosition(name, start.getVariablePosition(name));
            loaded_hand.update();
            collision_detection::CollisionResult loaded_self;
            if (!collisionFree(loaded_hand, loaded_self))
            {
              reason = "PREDICTED_SELF_COLLISION:" + pairText(loaded_self);
              return false;
            }
            retreat_contacts = worldContacts(loaded_hand);
            if (retreat_contacts.contact_count >= 200 ||
                (retreat_contacts.collision && retreat_contacts.contacts.empty()))
            {
              reason = "WORLD_CONTACT_DETAILS_UNAVAILABLE";
              return false;
            }
          }
          auto next_contacts = starting_contacts;
          for (auto& pair : next_contacts) pair.second = 0.0;
          for (const auto& pair : retreat_contacts.contacts)
          {
            const auto previous = previous_contacts.find(pair.first);
            double depth = 0.0;
            for (const auto& contact : pair.second) depth = std::max(depth, contact.depth);
            if (previous == previous_contacts.end() || depth > previous->second + 1e-7)
            {
              std::ostringstream detail;
              detail << "PREDICTED_WORLD_COLLISION:" << pair.first.first << "," << pair.first.second
                     << ":depth=" << depth << ":previous="
                     << (previous == previous_contacts.end() ? 0.0 : previous->second)
                     << ":sample=" << sample << "/" << samples;
              reason = detail.str();
              return false;
            }
            next_contacts[pair.first] = depth;
          }
          previous_contacts = next_contacts;
        }
      }
      for (const auto& pair : starting_contacts)
      {
        if (previous_contacts.at(pair.first) > std::max(0.0, pair.second - 1e-6))
        {
          reason = "CONTACT_NOT_SEPARATING:" + pair.first.first + "," + pair.first.second;
          return false;
        }
      }
      return true;
    };

    // Servo already checks its own arm command against its planning scene.
    // This strict ACM adds only the full arm/hand geometry Servo intentionally
    // exempts.  Evaluate one combined future (arm command plus the current
    // hand reference); extrapolating noisy measured finger velocities as
    // separate scenarios caused false joint-limit stops and hundreds of FCL
    // calls in a single 20 ms control period.
    std::map<std::string, double> combined_future;
    for (const std::string& name : required_joint_names_)
      combined_future[name] = 0.0;
    for (std::size_t index = 0; index < arm_joint_names_.size(); ++index)
      combined_future[arm_joint_names_[index]] = velocity[index];
    for (const auto& target : hand_reference)
    {
      const auto& bounds = model_->getVariableBounds(target.first);
      const double requested_rate =
          (target.second - start.getVariablePosition(target.first)) /
          prediction_horizon_s_;
      // A contact-preloaded finger cannot reach its empty-hand CLOSE target
      // instantaneously. Predict at its bounded commanded speed, so holding
      // an object does not invent a much more folded hand in the next 80 ms.
      const double maximum_rate = bounds.velocity_bounded_ ?
          bounds.max_velocity_ : std::abs(requested_rate);
      combined_future[target.first] =
          std::max(-maximum_rate, std::min(maximum_rate, requested_rate));
    }

    if (!check_scenario(combined_future, "STRICT_ARM_HAND_FUTURE"))
      return false;
    reason = "SAFE";
    return true;
  }

  void rawVelocityCallback(const std_msgs::Float64MultiArrayConstPtr& message)
  {
    if (constrained_velocity_projection_) return;
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
    double prediction_valid_until_wall_s = 0.0;
    std::map<std::string, double> hand_reference;
    std::map<std::string, double> arm_reference;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const double state_age = have_complete_state_ ?
          (ros::WallTime::now() - latest_state_wall_time_).toSec() :
          std::numeric_limits<double>::infinity();
      ready = have_complete_state_ && !fault_latched_ &&
              state_age <= state_timeout_s_;
      hand_motion_active = hand_motion_active_;
      hand_reference = latest_hand_reference_;
      arm_reference = latest_arm_reference_;
      prediction_valid_until_wall_s = latest_state_wall_time_.toSec() + prediction_horizon_s_;
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
    if (enable_swept_command_gate_ && !predictedVelocityIsSafe(
            velocity, start, hand_reference, arm_reference, reason))
    {
      publishZeroVelocity();
      publishCommandStatus(false, reason);
      ROS_WARN_STREAM_THROTTLE(
          0.5, "MoveIt/FCL blocked predicted arm command: " << reason
          << "; " << stateText(start));
      return;
    }
    std_msgs::Float64MultiArray safe = *message;
    // Computation and ROS delivery consume this horizon; receipt must not
    // restart its clock. The EGM integrator independently stops at expiry,
    // including when this checker is busy and cannot publish a fresh Bool.
    if (ros::WallTime::now().toSec() >= prediction_valid_until_wall_s)
    {
      publishZeroVelocity();
      publishCommandStatus(false, "PREDICTION_EXPIRED_BEFORE_PUBLICATION");
      return;
    }
    safe.data.push_back(prediction_valid_until_wall_s);
    safe_velocity_publisher_.publish(safe);
    publishCommandStatus(true, "SAFE");
  }

  void handCommandCallback(
      const trajectory_msgs::JointTrajectoryConstPtr& message)
  {
    if (message->points.empty())
    {
      ROS_ERROR_STREAM_THROTTLE(
          1.0, "Collision guard rejected empty hand trajectory");
      return;
    }
    const trajectory_msgs::JointTrajectoryPoint& point =
        message->points.back();
    if (point.positions.size() != message->joint_names.size())
    {
      ROS_ERROR_STREAM_THROTTLE(
          1.0, "Collision guard rejected malformed hand trajectory");
      return;
    }
    std::map<std::string, double> values;
    for (std::size_t index = 0; index < message->joint_names.size(); ++index)
    {
      if (!values.emplace(
              message->joint_names[index], point.positions[index]).second ||
          !std::isfinite(point.positions[index]))
      {
        ROS_ERROR_STREAM_THROTTLE(
            1.0, "Collision guard rejected invalid hand trajectory value");
        return;
      }
    }
    std::map<std::string, double> accepted;
    for (const std::string& name : hand_reference_joint_names_)
    {
      const auto value = values.find(name);
      if (value == values.end())
      {
        ROS_ERROR_STREAM_THROTTLE(
            1.0, "Collision guard rejected incomplete hand trajectory");
        return;
      }
      const moveit::core::JointModel* joint = model_->getJointOfVariable(name);
      if (!joint->satisfiesPositionBounds(&value->second, 0.0))
      {
        ROS_ERROR_STREAM_THROTTLE(
            1.0, "Collision guard rejected out-of-bounds hand trajectory");
        return;
      }
      accepted[name] = value->second;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      latest_hand_reference_ = accepted;
    }
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
    // complete state is accepted only while every required fragment is fresh.
    // A publisher pause makes the output fail closed until fresh fragments
    // arrive again; it is not a permanent robot fault by itself.
    if (message->name.size() != message->position.size())
    {
      latchFault("INVALID_JOINT_STATE:joint name/position length mismatch");
      return;
    }
    if (!message->velocity.empty() &&
        message->name.size() != message->velocity.size())
    {
      latchFault("INVALID_JOINT_STATE:joint name/velocity length mismatch");
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
      if (!message->velocity.empty() &&
          !std::isfinite(message->velocity[index]))
      {
        latchFault("INVALID_JOINT_STATE:non-finite joint velocity: " + name);
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
        if (!joint)
          continue;
        double measured_velocity = 0.0;
        const auto previous_position =
            latest_independent_positions_.find(item.first);
        const auto previous_stamp = latest_joint_wall_times_.find(item.first);
        if (!message->velocity.empty())
        {
          const auto message_index = std::find(
              message->name.begin(), message->name.end(), item.first) -
              message->name.begin();
          measured_velocity = message->velocity[message_index];
        }
        else if (previous_position != latest_independent_positions_.end() &&
                 previous_stamp != latest_joint_wall_times_.end())
        {
          const double dt = (received_at - previous_stamp->second).toSec();
          if (dt > 1.0e-4 && dt <= state_timeout_s_)
            measured_velocity = (item.second - previous_position->second) / dt;
        }
        latest_independent_positions_[item.first] = item.second;
        latest_independent_velocities_[item.first] = measured_velocity;
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
        const auto measured_velocity =
            latest_independent_velocities_.find(name);
        merged.velocity.push_back(
            measured_velocity == latest_independent_velocities_.end() ?
                0.0 : measured_velocity->second);
      }
      // Sensor observations of passive joints are not commands to those
      // joints. Preserve them after processing the independent variables.
      for (const auto* joint : model_->getMimicJointModels())
      {
        for (const auto& name : joint->getVariableNames())
        {
          const auto position = latest_independent_positions_.find(name);
          if (position == latest_independent_positions_.end()) continue;
          if ((received_at - latest_joint_wall_times_.at(name)).toSec() > state_timeout_s_) return;
          merged.name.push_back(name);
          merged.position.push_back(position->second);
          merged.velocity.push_back(latest_independent_velocities_.at(name));
        }
      }
    }
    std::string error;
    bool valid_positions = applyIndependentPositions(
            merged, candidate, true, measured_joint_limit_tolerance_rad_, error);
    {
      for (std::size_t i = 0; i < merged.name.size(); ++i)
        if (model_->getJointOfVariable(merged.name[i])->getMimic())
          candidate.setVariablePosition(merged.name[i], merged.position[i]);
      candidate.update();
      valid_positions = positionsSatisfyBounds(candidate, measured_joint_limit_tolerance_rad_, &error);
    }
    const bool permitted_hand_deflection = !valid_positions &&
        constrained_velocity_projection_ && recover_simulated_hand_limit_transients_ &&
        ros::Time::isSimTime() && onlyHandPositionLimitsViolated(candidate);
    if (!valid_positions && !permitted_hand_deflection)
    {
      // A compliant Gazebo finger can briefly deflect through its joint stop
      // under contact. The 11:27 recording had one such sample (-0.023995 rad),
      // then valid feedback again, but the permanent emergency latch blocked
      // every later arm command. Stop while the measured hand is invalid;
      // accept motion only after fresh, bounded, collision-free feedback.
      // Neither the position tolerance nor any real-robot fault is relaxed.
      if (recover_simulated_hand_limit_transients_ && ros::Time::isSimTime() &&
          onlyHandPositionLimitsViolated(candidate))
      {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          have_complete_state_ = false;
          transient_state_reason_ = "SIMULATED_HAND_LIMIT_HOLD:" + error;
        }
        publishZeroVelocity();
        publishCommandStatus(false, "SIMULATED_HAND_LIMIT_HOLD:" + error);
        return;
      }
      latchFault("INVALID_JOINT_STATE:" + error);
      return;
    }
    for (std::size_t index = 0; index < merged.name.size(); ++index)
      candidate.setVariableVelocity(merged.name[index], merged.velocity[index]);

    collision_detection::CollisionResult result;
    if (!collisionFree(candidate, result))
    {
      latchFault("SELF_COLLISION:" + pairText(result) + ":" +
                 stateText(candidate));
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      *latest_state_ = candidate;
      latest_state_wall_time_ = received_at;
      have_complete_state_ = true;
      transient_state_reason_ = permitted_hand_deflection ?
          "SIMULATED_HAND_DEFLECTION_ARM_GEOMETRY_VALID" : "";
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
    // The monitored topic normally publishes diffs. Obtain one complete
    // starting snapshot, then apply its subsequent diffs in the subscriber.
    if (geometric_command_filter_ && !have_scene_snapshot_ &&
        (ros::WallTime::now() - last_scene_request_).toSec() >= 1.0)
    {
      last_scene_request_ = ros::WallTime::now();
      if (planning_scene_client_.exists())
      {
        moveit_msgs::GetPlanningScene request;
        request.request.components.components = 1023;
        if (planning_scene_client_.call(request))
          have_scene_snapshot_ = scene_->usePlanningSceneMsg(request.response.scene);
      }
    }
    bool have_state;
    bool fault;
    double age;
    std::string detail;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      have_state = have_complete_state_;
      fault = fault_latched_;
      detail = fault_reason_;
      if (!fault && !have_state && !transient_state_reason_.empty())
        detail = transient_state_reason_;
      age = have_state ?
          (ros::WallTime::now() - latest_state_wall_time_).toSec() :
          std::numeric_limits<double>::infinity();
    }
    if (!fault && have_state && age > state_timeout_s_)
      detail = "JOINT_STATE_TIMEOUT";
    const bool safe = have_state && !fault && age <= state_timeout_s_;
    publishStatus(safe, safe ? "SAFE" :
        (detail.empty() ? "WAITING_FOR_COMPLETE_JOINT_STATE" : detail));
    // Malformed state, arm-limit and measured self-collision faults latch.
    // Opt-in simulated hand-stop deflection holds without a permanent latch.
    // A transient stale state still fails closed below, but automatically
    // recovers after the next complete fresh joint-state snapshot.
    if (fault)
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
      if (!safe)
        publishCommandStatus(false, detail.empty() ?
            "WAITING_FOR_COMPLETE_JOINT_STATE" : detail);
      else if (command_stale)
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
  std::map<std::string, double> latest_independent_velocities_;
  std::map<std::string, ros::WallTime> latest_joint_wall_times_;
  std::map<std::string, double> latest_hand_reference_;
  std::map<std::string, double> latest_arm_reference_;
  std::map<std::string, double> latest_reference_velocity_;
  std::map<std::string, double> reference_acceleration_;
  ros::Subscriber arm_reference_subscriber_;
  ros::Subscriber robot_zero_subscriber_;
  Eigen::Matrix3d robot_zero_rotation_{ Eigen::Matrix3d::Identity() };
  bool have_robot_zero_{ false };
  double maximum_side_angle_{ M_PI/2.0 };
  ros::WallTime latest_state_wall_time_;
  bool have_complete_state_{ false };
  bool fault_latched_{ false };
  bool recover_simulated_hand_limit_transients_{ false };
  std::string transient_state_reason_;
  std::string fault_reason_;
  std::string joint_state_topic_;
  std::string service_name_;
  std::string raw_velocity_topic_;
  std::string safe_velocity_topic_;
  std::string hand_motion_service_;
  std::string hand_command_topic_;
  std::vector<std::string> required_joint_names_;
  std::vector<std::string> arm_joint_names_;
  std::vector<std::string> hand_reference_joint_names_;
  std::vector<double> initial_hand_reference_;
  double state_timeout_s_{ 0.25 };
  double status_rate_hz_{ 50.0 };
  double command_timeout_s_{ 0.10 };
  bool enable_swept_command_gate_{ true };
  bool constrained_velocity_projection_{ false };
  double prediction_horizon_s_{ 0.40 };
  double maximum_prediction_step_rad_{ 0.01 };
  int maximum_prediction_samples_{ 256 };
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
  ros::Subscriber hand_command_subscriber_;
  ros::ServiceServer service_;
  ros::ServiceServer hand_motion_service_server_;
  ros::Timer status_timer_;
  bool geometric_command_filter_{ false };
  bool have_scene_snapshot_{ false };
  bool proximity_recovery_active_{ false };
  double proximity_enter_scale_{ 0.20 };
  double proximity_release_scale_{ 0.80 };
  double collision_scale_{ 1.0 };
  double inverse_damping_{ 0.0 };
  double singularity_soft_{ 17.0 };
  double singularity_hard_{ 30.0 };
  double joint_limit_margin_{ 0.08 };
  ros::Publisher geometric_command_publisher_;
  ros::Publisher geometric_diagnostic_publisher_;
  ros::Subscriber geometric_command_subscriber_;
  ros::Subscriber planning_scene_subscriber_;
  ros::Subscriber proximity_scale_subscriber_;
  ros::ServiceClient planning_scene_client_;
  ros::WallTime last_scene_request_;
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
