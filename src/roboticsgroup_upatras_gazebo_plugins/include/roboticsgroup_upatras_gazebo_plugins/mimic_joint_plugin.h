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

#ifndef ROBOTICSGROUP_UPATRAS_GAZEBO_PLUGINS_MIMIC_JOINT_PLUGIN
#define ROBOTICSGROUP_UPATRAS_GAZEBO_PLUGINS_MIMIC_JOINT_PLUGIN

// ROS includes
#include <ros/ros.h>

// ros_control
#include <control_toolbox/pid.h>

// Gazebo includes
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>

namespace gazebo {

    class MimicJointPlugin : public ModelPlugin {
      public:
        MimicJointPlugin();
        virtual ~MimicJointPlugin() override;

        virtual void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf) override;

      private:
        void UpdateChild();
        void UpdateDiagnostics();

        // Parameters
        std::string joint_name_, mimic_joint_name_, robot_namespace_;
        double multiplier_, offset_, sensitiveness_, max_effort_, force_sign_;
        double near_target_effort_, near_target_error_, max_velocity_;
        double diagnostic_velocity_threshold_, diagnostic_heartbeat_period_s_;
        bool has_pid_, diagnostic_velocity_above_;
        double last_raw_effort_, last_applied_effort_;
        bool last_effort_commanded_;
        unsigned long long diagnostic_update_count_;
        unsigned long long diagnostic_window_update_count_;
        double diagnostic_last_heartbeat_sim_time_s_;
        double diagnostic_window_start_sim_time_s_;
        double diagnostic_window_max_abs_source_velocity_;
        double diagnostic_window_max_abs_mimic_velocity_;

        // PID controller if needed
        control_toolbox::Pid pid_;

        // Event-only diagnostic. It never participates in the control law.
        ros::Publisher diagnostic_pub_;

        // Pointers to the joints
        physics::JointPtr joint_, mimic_joint_;

        // Pointer to the model
        physics::ModelPtr model_;

        // Pointer to the world
        physics::WorldPtr world_;

        // Pointer to the update event connection
        event::ConnectionPtr update_connection_;

        // Post-physics observation connection.  Keeping diagnostics separate
        // from the control callback prevents a pre-integration sample from
        // hiding a one-step constraint or contact velocity spike.
        event::ConnectionPtr diagnostic_connection_;
    };

}

#endif  // ROBOTICSGROUP_UPATRAS_GAZEBO_PLUGINS_MIMIC_JOINT_PLUGIN
