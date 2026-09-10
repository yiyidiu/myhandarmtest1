/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2013, Open Source Robotics Foundation
 *  Copyright (c) 2013, The Johns Hopkins University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Open Source Robotics Foundation
 *     nor the names of its contributors may be
 *     used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

// The Noetic default backend uses shortest_angular_distance_with_limits
// for every revolute joint and ignores its failure return. That routine is
// limited to a single-turn interval: IRB120 joint_6 has +/-400 deg limits,
// so at -0.70 -> -0.78 rad it returns +6.203 rad instead of -0.08 rad.
// Reuse its hardware interfaces, PID gains, unwrapped readSim feedback and
// joint-limit registration. Only the bounded position-PID write path changes.
#include <gazebo_ros_control/default_robot_hw_sim.h>
#include <pluginlib/class_list_macros.h>
#include <angles/angles.h>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <std_msgs/Float64MultiArray.h>

namespace handarm_sim_demo
{
class BoundedPositionHWSim : public gazebo_ros_control::DefaultRobotHWSim
{
public:
  bool initSim(const std::string& robot_namespace, ros::NodeHandle model_nh,
               gazebo::physics::ModelPtr model, const urdf::Model* const urdf_model,
               std::vector<transmission_interface::TransmissionInfo> transmissions) override
  {
    if (!DefaultRobotHWSim::initSim(robot_namespace, model_nh, model,
                                  urdf_model, transmissions)) return false;
    for (unsigned int j = 0; j < n_dof_; ++j)
    {
      if (joint_control_methods_[j] != POSITION_PID ||
          joint_types_[j] != urdf::Joint::REVOLUTE ||
          !std::isfinite(joint_lower_limits_[j]) ||
          !std::isfinite(joint_upper_limits_[j]) ||
          joint_lower_limits_[j] >= joint_upper_limits_[j])
      {
        ROS_ERROR_STREAM("BoundedPositionHWSim requires bounded revolute position-PID joints: "
                         << joint_names_[j]);
        return false;
      }
    }
    ros::param::param("/gazebo_ros_control/use_reference_velocity_derivative",
                      use_reference_velocity_derivative_, false);
    reference_velocity_.assign(n_dof_, 0.0);
    if (use_reference_velocity_derivative_)
    {
      // Near the straight wrist, explicit -D*qdot at a 1 ms step excited
      // opposite joint_4/joint_6 oscillations above 40 rad/s with a <1 rad/s
      // reference. Put measured-velocity damping in ODE's implicit joint
      // solve, which resolves the coupled wrist inertia in the same step.
      // The physical-profile URDF enables implicitSpringDamper on these
      // joints. P/I and the desired-velocity D feedforward remain below.
      // This mode owns total joint damping: it replaces the URDF damping
      // (joint_6 previously added 0.5) rather than applying D twice.
      for (unsigned int j = 0; j < n_dof_; ++j)
        sim_joints_[j]->SetStiffnessDamping(
            0, 0.0, pid_controllers_[j].getGains().d_gain_, 0.0);
      reference_subscriber_ = model_nh.subscribe<std_msgs::Float64MultiArray>(
          "/egm_position_reference/command_state", 1,
          [this](const std_msgs::Float64MultiArrayConstPtr& message) {
            if (message->data.size() != 2 * n_dof_) return;
            for (double value : message->data) if (!std::isfinite(value)) return;
            std::lock_guard<std::mutex> lock(reference_mutex_);
            std::copy(message->data.begin() + n_dof_, message->data.end(), reference_velocity_.begin());
            reference_received_ = ros::WallTime::now();
          });
    }
    ROS_INFO("Bounded revolute position PID enabled; multi-turn joint limits retained");
    return true;
  }

  void writeSim(ros::Time, ros::Duration period) override
  {
    // Keep the base backend's position-hold emergency stop semantics.
    if (e_stop_active_ && !last_e_stop_active_)
      last_joint_position_command_ = joint_position_;
    last_e_stop_active_ = e_stop_active_;
    if (e_stop_active_) joint_position_command_ = last_joint_position_command_;
    pj_sat_interface_.enforceLimits(period);
    pj_limits_interface_.enforceLimits(period);

    std::vector<double> reference_velocity(n_dof_, 0.0);
    if (use_reference_velocity_derivative_ && !e_stop_active_)
    {
      std::lock_guard<std::mutex> lock(reference_mutex_);
      if ((ros::WallTime::now() - reference_received_).toSec() < 0.05)
        reference_velocity = reference_velocity_;
    }
    for (unsigned int j = 0; j < n_dof_; ++j)
    {
      const double current = joint_position_[j];
      const double target = joint_position_command_[j];
      double error = target - current;
      if (joint_lower_limits_[j] >= -M_PI && joint_upper_limits_[j] <= M_PI)
      {
        // Preserve the established calculation for single-turn joints.
        double circular_error = error;
        if (angles::shortest_angular_distance_with_limits(
                current, target, joint_lower_limits_[j], joint_upper_limits_[j],
                circular_error)) error = circular_error;
      }
      // For a multi-turn bounded joint, readSim already unwraps feedback.
      // The command is bounded above, so its signed coordinate difference
      // is the correct error, including across +/-pi. No URDF range is cut.
      double effort = 0.0;
      if (std::isfinite(error))
      {
        // A 250 Hz position target is a staircase to the 1 kHz physics loop.
        // Differentiating that staircase produces derivative kicks. The
        // measured-velocity term is applied implicitly by the physics joint;
        // only desired-velocity feedforward belongs in this explicit drive.
        effort = use_reference_velocity_derivative_ ?
            pid_controllers_[j].computeCommand(error,
                reference_velocity[j], period) :
            pid_controllers_[j].computeCommand(error, period);
      }
      if (!std::isfinite(effort)) effort = 0.0;
      effort = std::max(-joint_effort_limits_[j],
                        std::min(joint_effort_limits_[j], effort));
      sim_joints_[j]->SetForce(0, effort);
    }
  }
private:
  bool use_reference_velocity_derivative_{false};
  std::mutex reference_mutex_;
  std::vector<double> reference_velocity_;
  ros::WallTime reference_received_;
  ros::Subscriber reference_subscriber_;
};
}  // namespace handarm_sim_demo
PLUGINLIB_EXPORT_CLASS(handarm_sim_demo::BoundedPositionHWSim,
                       gazebo_ros_control::RobotHWSim)
