#include <cmath>
#include <gazebo_ros_control/default_robot_hw_sim.h>
#include <pluginlib/class_list_macros.hpp>

namespace fr3_allegro_sim {

class BoundedRevoluteHWSim : public gazebo_ros_control::DefaultRobotHWSim {
 public:
  void readSim(ros::Time time, ros::Duration period) override {
    DefaultRobotHWSim::readSim(time, period);
    constexpr double full_turn = 6.2831853071795864769;
    for (std::size_t i = 0; i < joint_position_.size(); ++i) {
      const double lower = joint_lower_limits_[i];
      const double upper = joint_upper_limits_[i];
      double& position = joint_position_[i];
      if (joint_types_[i] != urdf::Joint::REVOLUTE || upper <= lower ||
          upper - lower >= full_turn || (lower <= position && position <= upper)) {
        continue;
      }
      // A first reading or a model reset can select the adjacent turn. A bounded
      // joint has just one branch in its travel interval. Preserve small overruns
      // on that branch so the standard limit and effort controllers still see them.
      const double midpoint = lower + (upper - lower) / 2.0;
      position = midpoint + std::remainder(position - midpoint, full_turn);
    }
  }
};

}  // namespace fr3_allegro_sim

PLUGINLIB_EXPORT_CLASS(fr3_allegro_sim::BoundedRevoluteHWSim, gazebo_ros_control::RobotHWSim)
