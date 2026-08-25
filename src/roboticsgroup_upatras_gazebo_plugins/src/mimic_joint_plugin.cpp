/**
Copyright (c) 2014, Konstantinos Chatzilygeroudis
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer
    in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**/

#include <roboticsgroup_upatras_gazebo_plugins/mimic_joint_plugin.h>

#include <algorithm>
#include <cmath>
#include <sstream>

#include <std_msgs/String.h>

#if GAZEBO_MAJOR_VERSION >= 8
namespace math = ignition::math;
#else
namespace math = gazebo::math;
#endif

namespace gazebo {

    MimicJointPlugin::MimicJointPlugin()
        : last_raw_effort_(0.0),
          last_applied_effort_(0.0),
          last_effort_commanded_(false),
          diagnostic_update_count_(0),
          diagnostic_window_update_count_(0),
          diagnostic_last_heartbeat_sim_time_s_(0.0),
          diagnostic_window_start_sim_time_s_(0.0),
          diagnostic_window_max_abs_source_velocity_(0.0),
          diagnostic_window_max_abs_mimic_velocity_(0.0)
    {
        joint_.reset();
        mimic_joint_.reset();
    }

    MimicJointPlugin::~MimicJointPlugin()
    {
        update_connection_.reset();
        diagnostic_connection_.reset();
    }

    void MimicJointPlugin::Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf)
    {
        model_ = _parent;
        world_ = model_->GetWorld();

        // Error message if the model couldn't be found
        if (!model_) {
            ROS_ERROR("Parent model is NULL! MimicJointPlugin could not be loaded.");
            return;
        }

        // Check that ROS has been initialized
        if (!ros::isInitialized()) {
            ROS_ERROR("A ROS node for Gazebo has not been initialized, unable to load plugin.");
            return;
        }

        // Check for robot namespace
        if (_sdf->HasElement("robotNamespace")) {
            robot_namespace_ = _sdf->GetElement("robotNamespace")->Get<std::string>();
        }
        ros::NodeHandle model_nh(robot_namespace_);

        // Check for joint element
        if (!_sdf->HasElement("joint")) {
            ROS_ERROR("No joint element present. MimicJointPlugin could not be loaded.");
            return;
        }

        joint_name_ = _sdf->GetElement("joint")->Get<std::string>();

        // Check for mimicJoint element
        if (!_sdf->HasElement("mimicJoint")) {
            ROS_ERROR("No mimicJoint element present. MimicJointPlugin could not be loaded.");
            return;
        }

        mimic_joint_name_ = _sdf->GetElement("mimicJoint")->Get<std::string>();

        // Check if PID controller wanted
        has_pid_ = _sdf->HasElement("hasPID");
        if (has_pid_) {
            std::string name = _sdf->GetElement("hasPID")->Get<std::string>();
            if (name.empty()) {
                name = "gazebo_ros_control/pid_gains/" + mimic_joint_name_;
            }
            const ros::NodeHandle nh(model_nh, name);
            pid_.init(nh);
        }

        // Check for multiplier element
        multiplier_ = 1.0;
        if (_sdf->HasElement("multiplier"))
            multiplier_ = _sdf->GetElement("multiplier")->Get<double>();

        // Check for offset element
        offset_ = 0.0;
        if (_sdf->HasElement("offset"))
            offset_ = _sdf->GetElement("offset")->Get<double>();

        // Check for sensitiveness element
        sensitiveness_ = 0.0;
        if (_sdf->HasElement("sensitiveness"))
            sensitiveness_ = _sdf->GetElement("sensitiveness")->Get<double>();

        force_sign_ = 1.0;
        if (_sdf->HasElement("forceSign"))
            force_sign_ = _sdf->GetElement("forceSign")->Get<double>();
        if (!std::isfinite(force_sign_) || fabs(force_sign_) != 1.0) {
            ROS_ERROR_STREAM("forceSign for mimic joint \"" << mimic_joint_name_
                             << "\" must be +1 or -1");
            return;
        }

        // Get pointers to joints
        joint_ = model_->GetJoint(joint_name_);
        if (!joint_) {
            ROS_ERROR_STREAM("No joint named \"" << joint_name_ << "\". MimicJointPlugin could not be loaded.");
            return;
        }
        mimic_joint_ = model_->GetJoint(mimic_joint_name_);
        if (!mimic_joint_) {
            ROS_ERROR_STREAM("No (mimic) joint named \"" << mimic_joint_name_ << "\". MimicJointPlugin could not be loaded.");
            return;
        }

        // Check for max effort
#if GAZEBO_MAJOR_VERSION > 2
        max_effort_ = mimic_joint_->GetEffortLimit(0);
#else
        max_effort_ = mimic_joint_->GetMaxForce(0);
#endif
        if (_sdf->HasElement("maxEffort")) {
            max_effort_ = _sdf->GetElement("maxEffort")->Get<double>();
        }

        near_target_effort_ = max_effort_;
        near_target_error_ = 0.0;
        if (_sdf->HasElement("nearTargetEffort"))
            near_target_effort_ = _sdf->GetElement("nearTargetEffort")->Get<double>();
        if (_sdf->HasElement("nearTargetError"))
            near_target_error_ = _sdf->GetElement("nearTargetError")->Get<double>();
        if (!std::isfinite(near_target_effort_) || near_target_effort_ <= 0.0 ||
            near_target_effort_ > max_effort_ || !std::isfinite(near_target_error_) ||
            near_target_error_ < 0.0) {
            ROS_ERROR_STREAM("Invalid near-target effort limiter for mimic joint \""
                             << mimic_joint_name_ << "\"");
            return;
        }

        max_velocity_ = 0.0;
        if (_sdf->HasElement("maxVelocity"))
            max_velocity_ = _sdf->GetElement("maxVelocity")->Get<double>();
        if (!std::isfinite(max_velocity_) || max_velocity_ < 0.0) {
            ROS_ERROR_STREAM("Invalid maxVelocity for mimic joint \""
                             << mimic_joint_name_ << "\"");
            return;
        }
        if (max_velocity_ > 0.0)
            mimic_joint_->SetVelocityLimit(0, max_velocity_);

        diagnostic_velocity_threshold_ = 0.0;
        if (_sdf->HasElement("diagnosticVelocityThreshold"))
            diagnostic_velocity_threshold_ =
                _sdf->GetElement("diagnosticVelocityThreshold")->Get<double>();
        if (!std::isfinite(diagnostic_velocity_threshold_) ||
            diagnostic_velocity_threshold_ < 0.0) {
            ROS_ERROR_STREAM("Invalid diagnosticVelocityThreshold for mimic joint \""
                             << mimic_joint_name_ << "\"");
            return;
        }
        diagnostic_velocity_above_ = false;
        diagnostic_heartbeat_period_s_ = 0.1;
        if (_sdf->HasElement("diagnosticHeartbeatPeriod"))
            diagnostic_heartbeat_period_s_ =
                _sdf->GetElement("diagnosticHeartbeatPeriod")->Get<double>();
        if (!std::isfinite(diagnostic_heartbeat_period_s_) ||
            diagnostic_heartbeat_period_s_ <= 0.0) {
            ROS_ERROR_STREAM("Invalid diagnosticHeartbeatPeriod for mimic joint \""
                             << mimic_joint_name_ << "\"");
            return;
        }
        if (diagnostic_velocity_threshold_ > 0.0) {
            diagnostic_pub_ = model_nh.advertise<std_msgs::String>(
                "handarm_sim_demo/mimic_diagnostics", 100, false);
            diagnostic_last_heartbeat_sim_time_s_ = world_->SimTime().Double();
            diagnostic_window_start_sim_time_s_ =
                diagnostic_last_heartbeat_sim_time_s_;
        }

        // Set max effort
        if (!has_pid_) {
#if GAZEBO_MAJOR_VERSION > 2
            mimic_joint_->SetParam("fmax", 0, max_effort_);
#else
            mimic_joint_->SetMaxForce(0, max_effort_);
#endif
        }

        // Listen to the update event. This event is broadcast every
        // simulation iteration.
        update_connection_ = event::Events::ConnectWorldUpdateBegin(
            boost::bind(&MimicJointPlugin::UpdateChild, this));
        if (diagnostic_velocity_threshold_ > 0.0) {
            diagnostic_connection_ = event::Events::ConnectWorldUpdateEnd(
                boost::bind(&MimicJointPlugin::UpdateDiagnostics, this));
        }

        // Output some confirmation
        ROS_INFO_STREAM("MimicJointPlugin loaded! Joint: \"" << joint_name_ << "\", Mimic joint: \"" << mimic_joint_name_ << "\""
                                                             << ", Multiplier: " << multiplier_ << ", Offset: " << offset_
                                                             << ", MaxEffort: " << max_effort_ << ", Sensitiveness: " << sensitiveness_
                                                             << ", ForceSign: " << force_sign_
                                                             << ", NearTargetEffort: " << near_target_effort_
                                                             << ", NearTargetError: " << near_target_error_
                                                             << ", MaxVelocity: " << max_velocity_
                                                             << ", DiagnosticVelocityThreshold: "
                                                             << diagnostic_velocity_threshold_
                                                             << ", DiagnosticHeartbeatPeriod: "
                                                             << diagnostic_heartbeat_period_s_);
    }

    void MimicJointPlugin::UpdateChild()
    {
#if GAZEBO_MAJOR_VERSION >= 8
        static ros::Duration period(world_->Physics()->GetMaxStepSize());
#else
        static ros::Duration period(world_->GetPhysicsEngine()->GetMaxStepSize());
#endif

        // Set mimic joint's angle based on joint's angle
#if GAZEBO_MAJOR_VERSION >= 8
        double angle = joint_->Position(0) * multiplier_ + offset_;
        double a = mimic_joint_->Position(0);
#else
        double angle = joint_->GetAngle(0).Radian() * multiplier_ + offset_;
        double a = mimic_joint_->GetAngle(0).Radian();
#endif

        const double source_velocity = joint_->GetVelocity(0);
        const double mimic_velocity = mimic_joint_->GetVelocity(0);
        double raw_effort = 0.0;
        double applied_effort = 0.0;
        bool effort_commanded = false;

        if (fabs(angle - a) >= sensitiveness_) {
            if (has_pid_) {
                if (a != a)
                    a = angle;
                double error = angle - a;
                // Use the simulator's joint rates directly.  Differentiating the
                // sampled position error made the very light distal finger links
                // chatter against their limits and occasionally appear at a
                // random oscillation phase during verification.
                double velocity_error =
                    source_velocity * multiplier_ - mimic_velocity;
                raw_effort = pid_.computeCommand(error, velocity_error, period);
                double effort_limit = max_effort_;
                // Near the target, limit only torque that accelerates the
                // lightweight link.  Keep the full finite torque available
                // when the command opposes its velocity, otherwise a fast
                // release cannot brake before rebounding from the joint stop.
                if (fabs(error) < near_target_error_ &&
                    (mimic_velocity == 0.0 ||
                     force_sign_ * raw_effort * mimic_velocity >= 0.0)) {
                    effort_limit = near_target_effort_;
                }
                applied_effort = math::clamp(
                    raw_effort,
                    -effort_limit, effort_limit);
                mimic_joint_->SetForce(0, force_sign_ * applied_effort);
                effort_commanded = true;
            }
            else {
#if GAZEBO_MAJOR_VERSION >= 9
                mimic_joint_->SetPosition(0, angle, true);
#elif GAZEBO_MAJOR_VERSION > 2
                ROS_WARN_ONCE("The mimic_joint plugin is using the Joint::SetPosition method without preserving the link velocity.");
                ROS_WARN_ONCE("As a result, gravity will not be simulated correctly for your model.");
                ROS_WARN_ONCE("Please set gazebo_pid parameters or upgrade to Gazebo 9.");
                ROS_WARN_ONCE("For details, see https://github.com/ros-simulation/gazebo_ros_pkgs/issues/612");
                mimic_joint_->SetPosition(0, angle);
#else
                mimic_joint_->SetAngle(0, math::Angle(angle));
#endif
            }
        }

        last_raw_effort_ = raw_effort;
        last_applied_effort_ = applied_effort;
        last_effort_commanded_ = effort_commanded;
    }

    void MimicJointPlugin::UpdateDiagnostics()
    {
#if GAZEBO_MAJOR_VERSION >= 8
        const double source_position = joint_->Position(0);
        const double mimic_position = mimic_joint_->Position(0);
#else
        const double source_position = joint_->GetAngle(0).Radian();
        const double mimic_position = mimic_joint_->GetAngle(0).Radian();
#endif
        const double target_position = source_position * multiplier_ + offset_;
        const double source_velocity = joint_->GetVelocity(0);
        const double mimic_velocity = mimic_joint_->GetVelocity(0);
        const double sim_time_s = world_->SimTime().Double();
        ++diagnostic_update_count_;
        ++diagnostic_window_update_count_;
        diagnostic_window_max_abs_source_velocity_ = std::max(
            diagnostic_window_max_abs_source_velocity_, fabs(source_velocity));
        diagnostic_window_max_abs_mimic_velocity_ = std::max(
            diagnostic_window_max_abs_mimic_velocity_, fabs(mimic_velocity));

        const bool diagnostic_violation =
            diagnostic_velocity_threshold_ > 0.0 &&
            (!std::isfinite(source_position) || !std::isfinite(mimic_position) ||
             !std::isfinite(source_velocity) || !std::isfinite(mimic_velocity) ||
             fabs(source_velocity) > diagnostic_velocity_threshold_ ||
             fabs(mimic_velocity) > diagnostic_velocity_threshold_);
        if (diagnostic_violation && !diagnostic_velocity_above_) {
            std::ostringstream stream;
            stream << "{\"joint\":\"" << joint_name_
                   << "\",\"type\":\"velocity_event"
                   << "\",\"mimic_joint\":\"" << mimic_joint_name_
                   << "\",\"sim_time_s\":" << sim_time_s
                   << ",\"source_position_rad\":" << source_position
                   << ",\"mimic_position_rad\":" << mimic_position
                   << ",\"source_velocity_rad_s\":" << source_velocity
                   << ",\"mimic_velocity_rad_s\":" << mimic_velocity
                   << ",\"position_error_rad\":"
                   << (target_position - mimic_position)
                   << ",\"raw_effort\":" << last_raw_effort_
                   << ",\"applied_effort\":" << last_applied_effort_
                   << ",\"effort_commanded\":"
                   << (last_effort_commanded_ ? "true" : "false") << "}";
            std_msgs::String message;
            message.data = stream.str();
            diagnostic_pub_.publish(message);
            // Motion normally exceeds the quiet-window threshold.  The
            // fail-closed test runner decides whether an event happened in a
            // settled observation window; avoid flooding Gazebo's console
            // with expected in-motion crossings here.
            ROS_DEBUG_STREAM("MimicJointPlugin velocity diagnostic: "
                             << message.data);
        }
        diagnostic_velocity_above_ = diagnostic_violation;

        if (sim_time_s - diagnostic_last_heartbeat_sim_time_s_ >=
            diagnostic_heartbeat_period_s_) {
            std::ostringstream stream;
            stream << "{\"joint\":\"" << joint_name_
                   << "\",\"type\":\"heartbeat"
                   << "\",\"mimic_joint\":\"" << mimic_joint_name_
                   << "\",\"window_start_sim_time_s\":"
                   << diagnostic_window_start_sim_time_s_
                   << ",\"sim_time_s\":" << sim_time_s
                   << ",\"window_update_count\":"
                   << diagnostic_window_update_count_
                   << ",\"total_update_count\":"
                   << diagnostic_update_count_
                   << ",\"max_abs_source_velocity_rad_s\":"
                   << diagnostic_window_max_abs_source_velocity_
                   << ",\"max_abs_mimic_velocity_rad_s\":"
                   << diagnostic_window_max_abs_mimic_velocity_ << "}";
            std_msgs::String message;
            message.data = stream.str();
            diagnostic_pub_.publish(message);
            diagnostic_last_heartbeat_sim_time_s_ = sim_time_s;
            diagnostic_window_start_sim_time_s_ = sim_time_s;
            diagnostic_window_update_count_ = 0;
            diagnostic_window_max_abs_source_velocity_ = 0.0;
            diagnostic_window_max_abs_mimic_velocity_ = 0.0;
        }
    }

    GZ_REGISTER_MODEL_PLUGIN(MimicJointPlugin);

}  // namespace gazebo
