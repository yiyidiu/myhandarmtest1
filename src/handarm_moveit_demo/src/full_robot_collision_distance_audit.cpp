#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <ros/topic.h>
#include <sensor_msgs/JointState.h>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/collision_detection/collision_env.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/conversions.h>

namespace
{
struct PairDistance
{
  double distance;
  std::string first;
  std::string second;
  Eigen::Vector3d first_point;
  Eigen::Vector3d second_point;
};

bool byDistance(const PairDistance& left, const PairDistance& right)
{
  return left.distance < right.distance;
}
}  // namespace

int main(int argc, char** argv)
{
  // Keep the deterministic node name so ROS private-parameter command-line
  // overrides (for example _joint_override_names:=...) resolve correctly.
  ros::init(argc, argv, "full_robot_collision_distance_audit");
  ros::NodeHandle private_nh("~");

  std::string robot_description;
  std::string joint_state_topic;
  double wait_timeout_s;
  double report_distance_m;
  double required_clearance_m;
  int maximum_pairs;
  std::vector<std::string> joint_override_names;
  std::vector<double> joint_override_positions;
  private_nh.param<std::string>("robot_description", robot_description,
                                "robot_description");
  private_nh.param<std::string>("joint_state_topic", joint_state_topic,
                                "/joint_states");
  private_nh.param("wait_timeout_s", wait_timeout_s, 10.0);
  private_nh.param("report_distance_m", report_distance_m, 0.10);
  private_nh.param("required_clearance_m", required_clearance_m, 0.01);
  private_nh.param("maximum_pairs", maximum_pairs, 40);
  private_nh.getParam("joint_override_names", joint_override_names);
  private_nh.getParam("joint_override_positions", joint_override_positions);

  if (wait_timeout_s <= 0.0 || report_distance_m <= 0.0 ||
      required_clearance_m < 0.0 || maximum_pairs <= 0)
  {
    ROS_ERROR("Invalid collision-distance audit parameters");
    return 2;
  }
  if (joint_override_names.size() != joint_override_positions.size())
  {
    ROS_ERROR("joint_override_names and joint_override_positions must have equal lengths");
    return 2;
  }

  robot_model_loader::RobotModelLoader loader(robot_description);
  moveit::core::RobotModelPtr model = loader.getModel();
  if (!model)
  {
    ROS_ERROR_STREAM("Unable to load MoveIt model from parameter "
                     << robot_description);
    return 3;
  }

  const sensor_msgs::JointStateConstPtr joint_state =
      ros::topic::waitForMessage<sensor_msgs::JointState>(
          joint_state_topic, ros::Duration(wait_timeout_s));
  if (!joint_state)
  {
    ROS_ERROR_STREAM("No JointState received on " << joint_state_topic
                                                   << " within "
                                                   << wait_timeout_s << " s");
    return 4;
  }

  planning_scene::PlanningScene scene(model);
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  if (!moveit::core::jointStateToRobotState(*joint_state, state))
  {
    ROS_ERROR("JointState could not be converted to the MoveIt robot state");
    return 5;
  }
  const std::vector<std::string>& variables = model->getVariableNames();
  for (std::size_t index = 0; index < joint_override_names.size(); ++index)
  {
    const std::string& name = joint_override_names[index];
    const double position = joint_override_positions[index];
    if (std::find(variables.begin(), variables.end(), name) == variables.end() ||
        !std::isfinite(position))
    {
      ROS_ERROR_STREAM("Invalid joint override " << name << '=' << position);
      return 6;
    }
    const moveit::core::VariableBounds& bounds = model->getVariableBounds(name);
    if (bounds.position_bounded_ &&
        (position < bounds.min_position_ || position > bounds.max_position_))
    {
      ROS_ERROR_STREAM("Joint override " << name << '=' << position
                       << " violates [" << bounds.min_position_ << ", "
                       << bounds.max_position_ << ']');
      return 7;
    }
    state.setVariablePosition(name, position);
  }
  state.update();

  collision_detection::CollisionRequest collision_request;
  collision_detection::CollisionResult collision_result;
  collision_request.contacts = true;
  collision_request.max_contacts = 1000;
  collision_request.max_contacts_per_pair = 100;
  scene.checkSelfCollision(collision_request, collision_result, state);

  collision_detection::DistanceRequest distance_request;
  collision_detection::DistanceResult distance_result;
  distance_request.acm = &scene.getAllowedCollisionMatrix();
  distance_request.type = collision_detection::DistanceRequestTypes::ALL;
  distance_request.enable_nearest_points = true;
  distance_request.enable_signed_distance = true;
  distance_request.distance_threshold = report_distance_m;
  distance_request.max_contacts_per_body = 100;
  scene.getCollisionEnv()->distanceSelf(distance_request, distance_result,
                                        state);

  std::vector<PairDistance> distances;
  for (const auto& item : distance_result.distances)
  {
    for (const collision_detection::DistanceResultsData& data : item.second)
    {
      PairDistance row;
      row.distance = data.distance;
      row.first = data.link_names[0];
      row.second = data.link_names[1];
      row.first_point = data.nearest_points[0];
      row.second_point = data.nearest_points[1];
      distances.push_back(row);
    }
  }
  std::sort(distances.begin(), distances.end(), byDistance);

  const collision_detection::DistanceResultsData& minimum =
      distance_result.minimum_distance;
  std::cout << std::fixed << std::setprecision(9);
  std::cout << "SELF_COLLISION="
            << (collision_result.collision ? "true" : "false") << '\n';
  std::cout << "MINIMUM_DISTANCE_M=" << minimum.distance << '\n';
  std::cout << "MINIMUM_PAIR=" << minimum.link_names[0] << ','
            << minimum.link_names[1] << '\n';
  std::cout << "REQUIRED_CLEARANCE_M=" << required_clearance_m << '\n';
  for (std::size_t index = 0; index < joint_override_names.size(); ++index)
    std::cout << "JOINT_OVERRIDE=" << joint_override_names[index] << ','
              << joint_override_positions[index] << '\n';
  std::cout << "CLEARANCE_OK="
            << ((!collision_result.collision &&
                 minimum.distance >= required_clearance_m) ? "true" : "false")
            << '\n';
  std::cout << "PAIR_DISTANCE_M,LINK_1,LINK_2,POINT_1_XYZ,POINT_2_XYZ\n";
  const std::size_t rows = std::min(
      distances.size(), static_cast<std::size_t>(maximum_pairs));
  for (std::size_t index = 0; index < rows; ++index)
  {
    const PairDistance& row = distances[index];
    std::cout << row.distance << ',' << row.first << ',' << row.second << ','
              << row.first_point.transpose() << ','
              << row.second_point.transpose() << '\n';
  }

  if (collision_result.collision)
    return 10;
  if (minimum.distance < required_clearance_m)
    return 11;
  return 0;
}
