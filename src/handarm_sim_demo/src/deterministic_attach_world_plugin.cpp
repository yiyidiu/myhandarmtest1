#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <boost/bind/bind.hpp>
#include <gazebo/common/Events.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ros/advertise_service_options.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <std_srvs/SetBool.h>

namespace gazebo
{
class DeterministicAttachWorldPlugin : public WorldPlugin
{
public:
  DeterministicAttachWorldPlugin() = default;

  ~DeterministicAttachWorldPlugin() override
  {
    alive_.store(false);
    queue_.disable();
    service_.shutdown();
    if (queue_thread_.joinable())
      queue_thread_.join();
    update_connection_.reset();
  }

  void Load(physics::WorldPtr world, sdf::ElementPtr sdf) override
  {
    world_ = world;
    parent_model_name_ = GetString(sdf, "parentModel", "robot");
    parent_link_name_ = GetString(sdf, "parentLink", "link_6");
    child_model_name_ = GetString(sdf, "childModel", "target_object");
    child_link_name_ = GetString(sdf, "childLink", "object_link");
    service_name_ = GetString(
        sdf, "serviceName", "/handarm_sim_demo/set_object_attached");

    if (!ros::isInitialized())
    {
      gzerr << "DeterministicAttachWorldPlugin requires gazebo_ros_api_plugin\n";
      return;
    }

    node_.reset(new ros::NodeHandle(""));
    ros::AdvertiseServiceOptions options =
        ros::AdvertiseServiceOptions::create<std_srvs::SetBool>(
            service_name_,
            boost::bind(
                &DeterministicAttachWorldPlugin::OnRequest, this,
                boost::placeholders::_1, boost::placeholders::_2),
            ros::VoidPtr(), &queue_);
    service_ = node_->advertiseService(options);
    update_connection_ = event::Events::ConnectWorldUpdateBegin(
        std::bind(&DeterministicAttachWorldPlugin::OnUpdate, this));
    alive_.store(true);
    queue_thread_ = std::thread(
        &DeterministicAttachWorldPlugin::QueueThread, this);

    ROS_INFO_STREAM("Deterministic simulation attachment ready on "
                    << service_name_ << ": " << parent_model_name_ << "::"
                    << parent_link_name_ << " -> " << child_model_name_ << "::"
                    << child_link_name_);
  }

private:
  static std::string GetString(
      const sdf::ElementPtr &sdf, const std::string &name,
      const std::string &fallback)
  {
    return sdf->HasElement(name)
               ? sdf->GetElement(name)->Get<std::string>()
               : fallback;
  }

  bool OnRequest(
      std_srvs::SetBool::Request &request,
      std_srvs::SetBool::Response &response)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (pending_)
    {
      response.success = false;
      response.message = "another attachment request is pending";
      return true;
    }
    requested_attached_ = request.data;
    pending_ = true;
    completed_ = false;
    condition_.wait_for(lock, std::chrono::seconds(3), [this]() {
      return completed_ || !alive_.load();
    });
    if (!completed_)
    {
      pending_ = false;
      response.success = false;
      response.message = "Gazebo update thread did not process attachment request";
      return true;
    }
    response.success = operation_success_;
    response.message = operation_message_;
    return true;
  }

  void OnUpdate()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pending_)
      return;

    operation_success_ = requested_attached_ ? Attach() : Detach();
    completed_ = true;
    pending_ = false;
    condition_.notify_all();
  }

  bool Attach()
  {
    if (joint_)
    {
      operation_message_ = "object is already attached";
      return true;
    }
    const physics::ModelPtr parent_model = world_->ModelByName(parent_model_name_);
    const physics::ModelPtr child_model = world_->ModelByName(child_model_name_);
    if (!parent_model || !child_model)
    {
      operation_message_ = "parent or child model was not found";
      return false;
    }
    const physics::LinkPtr parent_link = parent_model->GetLink(parent_link_name_);
    const physics::LinkPtr child_link = child_model->GetLink(child_link_name_);
    if (!parent_link || !child_link)
    {
      operation_message_ = "parent or child link was not found";
      return false;
    }

    try
    {
      joint_ = parent_model->CreateJoint(
          joint_name_, "fixed", parent_link, child_link);
      if (!joint_)
      {
        operation_message_ = "Gazebo failed to create fixed joint";
        return false;
      }
      joint_->Init();
      operation_message_ = "simulation-only fixed attachment created";
      ROS_INFO_STREAM(operation_message_ << " between "
                      << parent_model_name_ << "::" << parent_link_name_ << " and "
                      << child_model_name_ << "::" << child_link_name_);
      return true;
    }
    catch (const std::exception &error)
    {
      joint_.reset();
      operation_message_ = std::string("fixed joint creation failed: ") + error.what();
      return false;
    }
  }

  bool Detach()
  {
    if (!joint_)
    {
      operation_message_ = "object is already detached";
      return true;
    }
    try
    {
      joint_->Detach();
      const physics::ModelPtr parent_model = world_->ModelByName(parent_model_name_);
      if (parent_model)
        parent_model->RemoveJoint(joint_name_);
      joint_.reset();
      operation_message_ = "simulation-only fixed attachment removed";
      return true;
    }
    catch (const std::exception &error)
    {
      operation_message_ = std::string("fixed joint removal failed: ") + error.what();
      return false;
    }
  }

  void QueueThread()
  {
    while (alive_.load() && ros::ok())
      queue_.callAvailable(ros::WallDuration(0.01));
  }

  physics::WorldPtr world_;
  physics::JointPtr joint_;
  event::ConnectionPtr update_connection_;
  std::unique_ptr<ros::NodeHandle> node_;
  ros::ServiceServer service_;
  ros::CallbackQueue queue_;
  std::thread queue_thread_;
  std::atomic<bool> alive_{false};
  std::mutex mutex_;
  std::condition_variable condition_;
  bool pending_{false};
  bool completed_{false};
  bool requested_attached_{false};
  bool operation_success_{false};
  std::string operation_message_;
  std::string service_name_;
  std::string parent_model_name_;
  std::string parent_link_name_;
  std::string child_model_name_;
  std::string child_link_name_;
  const std::string joint_name_{"handarm_demo_object_fixed_joint"};
};

GZ_REGISTER_WORLD_PLUGIN(DeterministicAttachWorldPlugin)
}  // namespace gazebo
