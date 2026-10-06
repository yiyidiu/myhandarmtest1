// Gazebo research plant for an external joint-speed interface.
// No externally integrated joint-position target. Not a calibrated ABB servo.
// The same URDF masses, gravity, contacts and joint effort limits are retained.
#include <gazebo_ros_control/default_robot_hw_sim.h>
#include <pluginlib/class_list_macros.h>
#include <algorithm>
#include <cmath>
#include <std_msgs/Float64MultiArray.h>

namespace handarm_sim_demo
{
class DirectVelocityHWSim : public gazebo_ros_control::DefaultRobotHWSim
{
public:
  bool initSim(const std::string& robot_namespace, ros::NodeHandle model_nh,
               gazebo::physics::ModelPtr model, const urdf::Model* const urdf_model,
               std::vector<transmission_interface::TransmissionInfo> transmissions) override
  {
    if (!DefaultRobotHWSim::initSim(robot_namespace, model_nh, model,
                                  urdf_model, transmissions)) return false;
    if (!ros::param::get("/gazebo_ros_control/direct_velocity_gains", gains_) ||
        gains_.size() != n_dof_) return false;
    world_ = model->GetWorld();
    descendants_.resize(n_dof_);
    raw_velocity_.assign(n_dof_, 0.0);
    observed_velocity_.assign(n_dof_, 0.0);
    integral_effort_.assign(n_dof_, 0.0);
    for (unsigned int j = 0; j < n_dof_; ++j)
    {
      if (joint_control_methods_[j] != VELOCITY_PID || !std::isfinite(gains_[j]) || gains_[j] <= 0)
        return false;
      // Resolve damping implicitly, as in the common position-profile plant.
      // This is a viscously damped finite-drive model, not an ideal rate setter.
      sim_joints_[j]->SetStiffnessDamping(0, 0.0, gains_[j], 0.0);
      for (const auto& link : model->GetLinks())
      {
        auto ancestor = urdf_model->getLink(link->GetName());
        while (ancestor && ancestor->parent_joint)
        {
          if (ancestor->parent_joint->name == joint_names_[j])
          { descendants_[j].push_back(link); break; }
          ancestor = ancestor->getParent();
        }
      }
    }
    diagnostic_ = model_nh.advertise<std_msgs::Float64MultiArray>(
        "/direct_velocity/plant_state", 1);
    ROS_WARN("Direct velocity research plant: gravity compensation, implicit rate damping; no q_ref");
    return true;
  }

  void readSim(ros::Time now, ros::Duration period) override
  {
    DefaultRobotHWSim::readSim(now, period);
    raw_velocity_ = joint_velocity_;
    const double dt = (now - last_read_).toSec();
    if (previous_position_.size() == n_dof_ && dt > 0.0 && dt < 0.02)
    {
      const double alpha = dt / (0.01 + dt);
      for (unsigned int j = 0; j < n_dof_; ++j)
        observed_velocity_[j] += alpha * (
            (joint_position_[j] - previous_position_[j]) / dt - observed_velocity_[j]);
    }
    previous_position_ = joint_position_;
    last_read_ = now;
    // ODE velocity readback omits some constraint position-correction motion.
    // Close and report the rate loop on actual angular displacement instead.
    joint_velocity_ = observed_velocity_;
  }

  void writeSim(ros::Time now, ros::Duration period) override
  {
    vj_sat_interface_.enforceLimits(period);
    vj_limits_interface_.enforceLimits(period);
    std_msgs::Float64MultiArray diag;
    diag.data.push_back(now.toSec());
    const auto gravity = world_->Gravity();
    for (unsigned int j = 0; j < n_dof_; ++j)
    {
      const double requested = e_stop_active_ ? 0.0 : joint_velocity_command_[j];
      const auto axis = sim_joints_[j]->GlobalAxis(0);
      const auto pivot = sim_joints_[j]->Anchor(0);
      double compensation = 0.0;
      for (const auto& link : descendants_[j])
      {
        const auto arm = link->WorldCoGPose().Pos() - pivot;
        compensation -= axis.Dot(arm.Cross(gravity * link->GetInertial()->Mass()));
      }
      // The measured-speed term is supplied by the implicit joint damper.
      const double speed_error = requested - observed_velocity_[j];
      const double candidate_integral = std::max(-0.25 * joint_effort_limits_[j],
          std::min(0.25 * joint_effort_limits_[j], integral_effort_[j] +
              5.0 * gains_[j] * speed_error * period.toSec()));
      const double proportional = compensation + gains_[j] *
          (requested + raw_velocity_[j] - observed_velocity_[j]);
      if (std::abs(proportional + candidate_integral) <= joint_effort_limits_[j] ||
          speed_error * (proportional + candidate_integral) < 0.0)
        integral_effort_[j] = candidate_integral;
      const double feed = proportional + integral_effort_[j];
      const double applied = std::isfinite(feed) ?
          std::max(-joint_effort_limits_[j], std::min(joint_effort_limits_[j], feed)) : 0.0;
      sim_joints_[j]->SetForce(0, applied);
      // Per joint: command, measured speed, position, gravity, explicit drive, drive limit.
      diag.data.insert(diag.data.end(), {requested, joint_velocity_[j], joint_position_[j],
                                       compensation, applied, joint_effort_limits_[j]});
    }
    diag.data.insert(diag.data.end(), raw_velocity_.begin(), raw_velocity_.end());
    if ((now - last_diagnostic_).toSec() >= 0.004)
    { diagnostic_.publish(diag); last_diagnostic_ = now; }
  }
private:
  gazebo::physics::WorldPtr world_;
  std::vector<double> gains_, raw_velocity_, observed_velocity_, previous_position_, integral_effort_;
  ros::Time last_read_;
  std::vector<std::vector<gazebo::physics::LinkPtr>> descendants_;
  ros::Publisher diagnostic_;
  ros::Time last_diagnostic_;
};
}
PLUGINLIB_EXPORT_CLASS(handarm_sim_demo::DirectVelocityHWSim, gazebo_ros_control::RobotHWSim)
