// Scene-only pouring illustration. Conserves a scalar water volume and draws
// a ballistic stream; it does not simulate liquid forces, slosh or grasping.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>

#include <gazebo/common/Events.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <std_msgs/String.h>

namespace gazebo
{
class PouringVisualizationPlugin : public WorldPlugin
{
  using Vec = ignition::math::Vector3d;
  using Pose = ignition::math::Pose3d;
  static constexpr double kPi = 3.14159265358979323846;
  static constexpr double kG = 9.81;
  static constexpr int kSegments = 12;
  // Geometry matches handarm_pouring.world, in metres and cubic metres.
  static constexpr double kBottleRadius = 0.030;
  static constexpr double kBottleBottom = -0.1115;
  static constexpr double kBottleTop = 0.068;
  static constexpr double kMouthHeight = 0.1005;
  static constexpr double kMouthRadius = 0.011;
  static constexpr double kCupRadius = 0.036;
  static constexpr double kCupFloor = -0.056;
  static constexpr double kCupRim = 0.060;
  static constexpr double kStreamRadius = 0.0025;
  // The visual stream radius is not the bottle's hydraulic outlet radius.
  // Keep the trajectory speed independent of the changing volume flow rate.
  static constexpr double kNominalStreamSpeed = 0.35;
  static constexpr double kMinimumDrainHead = 0.002;
  static constexpr double kPourTiltCosThreshold = 0.72;

public:
  ~PouringVisualizationPlugin() override
  {
    update_.reset();
    queue_.disable();
    reset_sub_.shutdown();
  }

  void Load(physics::WorldPtr world, sdf::ElementPtr sdf) override
  {
    world_ = world;
    if (!ros::isInitialized())
    {
      gzerr << "Pouring visualization requires gazebo_ros_api_plugin\n";
      return;
    }
    initial_ = sdf->HasElement("initialWaterMl")
        ? sdf->Get<double>("initialWaterMl") * 1e-6 : 94e-6;
    initial_ = std::max(0.0, std::min(initial_,
        kPi * kBottleRadius * kBottleRadius * (kBottleTop - kBottleBottom)));
    bottle_water_ = initial_;
    ros_node_.reset(new ros::NodeHandle());
    ros_node_->setCallbackQueue(&queue_);
    reset_sub_ = ros_node_->subscribe("/teleop_observers/object_reset_events", 1,
        &PouringVisualizationPlugin::OnReset, this);
    state_pub_ = ros_node_->advertise<std_msgs::String>("/teleop_pouring/state", 1);
    transport_node_.reset(new transport::Node());
    transport_node_->Init(world_->Name());
    visual_pub_ = transport_node_->Advertise<msgs::Visual>("~/visual", 100);
    ray_ = boost::dynamic_pointer_cast<physics::RayShape>(
        world_->Physics()->CreateShape("ray", physics::CollisionPtr()));
    update_ = event::Events::ConnectWorldUpdateBegin(
        std::bind(&PouringVisualizationPlugin::OnUpdate, this));
    ROS_INFO("倒水显示已启动：按瓶口姿态和落点计算接水，初始 %.0f ml；不模拟液体力。",
             initial_ * 1e6);
  }

private:
  void OnReset(const std_msgs::String::ConstPtr &msg)
  {
    std::string s = msg->data;
    s.erase(std::remove_if(s.begin(), s.end(),
        [](unsigned char c) { return std::isspace(c); }), s.end());
    if (s.find("\"success\":true") != std::string::npos)
      reset_requested_ = true;
  }

  // Volume of a cylindrical reservoir below a horizontal free surface.
  // Circle segments across 32 axial slices distinguish a mostly empty upright
  // bottle from one whose lower lip is actually below the water surface.
  static double VolumeBelow(const Pose &p, double radius, double low,
                            double high, double surface_z)
  {
    const Vec axis = p.Rot().RotateVector(Vec::UnitZ);
    const double a = std::sqrt(std::max(0.0, 1.0 - axis.Z() * axis.Z()));
    if (a < 1e-7)
    {
      const double cut = (surface_z - p.Pos().Z()) / axis.Z();
      const double length = axis.Z() > 0.0
          ? std::max(0.0, std::min(high, cut) - low)
          : std::max(0.0, high - std::max(low, cut));
      return kPi * radius * radius * length;
    }
    const double dz = (high - low) / 32.0;
    double volume = 0.0;
    for (int i = 0; i < 32; ++i)
    {
      const double z = low + (i + 0.5) * dz;
      const double u = (surface_z - p.Pos().Z() - axis.Z() * z) / (radius * a);
      if (u >= 1.0)
        volume += kPi * radius * radius * dz;
      else if (u > -1.0)
        volume += radius * radius * (std::acos(-u) + u * std::sqrt(1 - u*u)) * dz;
    }
    return volume;
  }

  static double WaterSurface(const Pose &p, double volume)
  {
    double low = p.Pos().Z() - 0.2, high = p.Pos().Z() + 0.2;
    for (int i = 0; i < 20; ++i)
    {
      const double mid = (low + high) * 0.5;
      if (VolumeBelow(p, kBottleRadius, kBottleBottom, kBottleTop, mid) < volume)
        low = mid;
      else
        high = mid;
    }
    return (low + high) * 0.5;
  }

  static Vec StreamPoint(const Vec &p, const Vec &v, double t)
  {
    return p + v*t + Vec(0, 0, -0.5*kG*t*t);
  }

  // Aim a ballistic stream through the center of the cup opening. With a
  // stationary bottle and cup this produces a stationary arc; only the visible
  // endpoint moves upward as the cup fills. Never infer exit speed from flow
  // rate divided by the rendered cylinder area: that makes the arc drift as
  // the bottle empties because the rendered radius is not a hydraulic outlet.
  static Vec StableStreamVelocity(const Vec &origin, const Vec &axis,
                                  const Pose &cup)
  {
    const Vec cup_axis = cup.Rot().RotateVector(Vec::UnitZ);
    const Vec target = cup.Pos() + cup_axis*kCupRim;
    const double drop = target.Z() - origin.Z();
    const double vertical_speed = axis.Z() * kNominalStreamSpeed;
    if (drop >= -1e-4)
      return axis * kNominalStreamSpeed;

    const double discriminant =
        vertical_speed*vertical_speed - 2.0*kG*drop;
    if (discriminant <= 0.0)
      return axis * kNominalStreamSpeed;

    const double flight_time =
        (vertical_speed + std::sqrt(discriminant))/kG;
    if (flight_time <= 1e-4)
      return axis * kNominalStreamSpeed;

    const double dx = target.X() - origin.X();
    const double dy = target.Y() - origin.Y();
    const double horizontal_distance = std::hypot(dx, dy);
    const Vec axis_horizontal(axis.X(), axis.Y(), 0.0);
    const Vec target_horizontal(dx, dy, 0.0);
    // If the cup is too far away or lies behind the bottle's pour direction,
    // keep the ordinary bottle-axis trajectory rather than visibly bending
    // water toward an unrelated target.
    if (horizontal_distance > 0.35 ||
        (horizontal_distance > 1e-4 && axis_horizontal.Length() > 1e-4 &&
         axis_horizontal.Normalized().Dot(target_horizontal.Normalized()) < 0.25))
      return axis * kNominalStreamSpeed;

    const Vec velocity(dx/flight_time, dy/flight_time, vertical_speed);
    if (!std::isfinite(velocity.Length()) || velocity.Length() > 2.0)
      return axis * kNominalStreamSpeed;
    return velocity;
  }

  // First downward crossing of a plane, including tilted cup rims.
  static double Crossing(const Vec &p, const Vec &v,
                         const Vec &plane, const Vec &normal)
  {
    const double a = -0.5*kG*normal.Z();
    const double b = normal.Dot(v), c = normal.Dot(p - plane);
    double answer = -1.0;
    if (std::abs(a) < 1e-8)
      return b < -1e-8 && -c/b > 0.0 ? -c/b : -1.0;
    const double disc = b*b - 4*a*c;
    if (disc < 0.0)
      return -1.0;
    for (const double t : {(-b + std::sqrt(disc))/(2*a),
                           (-b - std::sqrt(disc))/(2*a)})
      if (t > 0.0 && 2*a*t + b < 0.0 && (answer < 0.0 || t < answer))
        answer = t;
    return answer;
  }

  void DrawCylinder(const std::string &name, const Pose &pose,
                    double radius, double length, bool visible, double alpha)
  {
    msgs::Visual msg;
    msg.set_name("pouring_water::water_link::" + name);
    msg.set_parent_name("pouring_water::water_link");
    msgs::Set(msg.mutable_pose(), pose);
    msg.set_visible(visible);
    msg.set_transparency(alpha);
    msg.set_cast_shadows(false);
    // Geometry updates can recreate the Ogre material; send the water colour
    // too, otherwise camera views may render the moving stream default grey.
    auto *material = msg.mutable_material();
    material->set_lighting(true);
    for (auto *colour : {material->mutable_ambient(), material->mutable_diffuse()})
    {
      colour->set_r(0.06); colour->set_g(0.38);
      colour->set_b(0.95); colour->set_a(1.0);
    }
    auto *emissive = material->mutable_emissive();
    emissive->set_r(0.015); emissive->set_g(0.035);
    emissive->set_b(0.08); emissive->set_a(1.0);
    msg.mutable_geometry()->set_type(msgs::Geometry::CYLINDER);
    auto *cylinder = msg.mutable_geometry()->mutable_cylinder();
    cylinder->set_radius(radius);
    cylinder->set_length(std::max(0.00001, length));
    visual_pub_->Publish(msg);
  }

  void OnUpdate()
  {
    const double now = world_->SimTime().Double();
    if (now < last_update_)
      reset_requested_ = true;
    if (!reset_requested_ && now - last_update_ < 0.05)
      return;
    const double dt = std::max(0.0, std::min(0.1, now - last_update_));
    last_update_ = now;
    queue_.callAvailable(ros::WallDuration(0));
    if (reset_requested_)
    {
      bottle_water_ = initial_;
      cup_water_ = spilled_ = 0.0;
      reset_requested_ = false;
    }
    const auto bottle = world_->ModelByName("ground_object_left");
    const auto cup = world_->ModelByName("target_object");
    if (!bottle || !cup || !world_->ModelByName("pouring_water"))
      return;
    const Pose bp = bottle->WorldPose(), cp = cup->WorldPose();
    const Vec ba = bp.Rot().RotateVector(Vec::UnitZ);
    const Vec ca = cp.Rot().RotateVector(Vec::UnitZ);
    const Vec mouth = bp.Pos() + ba*kMouthHeight;
    const Vec down_rim = Vec::UnitZ - ba*ba.Z();
    const Vec lip = mouth - (down_rim.Length() > 1e-8
        ? down_rim.Normalized()*kMouthRadius : Vec::Zero);
    const double head = bottle_water_ > 1e-9
        ? WaterSurface(bp, bottle_water_) - lip.Z() : 0.0;
    // Illustration parameter, not a calibrated bottle-flow model. Once the
    // bottle is tipped, retain a small trickle head so the simplified reservoir
    // drains completely instead of stopping with invisible residual water.
    const bool tipped_for_drain = ba.Z() < kPourTiltCosThreshold;
    const double effective_head = std::max(head, kMinimumDrainHead);
    double rate = bottle_water_ > 1e-9 && (head > 0.0 || tipped_for_drain)
        ? std::min(40e-6, 0.55*kPi*0.003*0.003*
            std::sqrt(2*kG*effective_head)) : 0.0;
    rate = std::min(rate, dt > 0.0 ? bottle_water_/dt : 0.0);
    bool receiving = false;
    Vec velocity = Vec::Zero;
    double duration = 0.0;
    if (rate > 1e-9)
    {
      velocity = StableStreamVelocity(lip, ba, cp);
      duration = Crossing(lip, velocity, Vec::Zero, Vec::UnitZ);
      if (duration <= 0.0) duration = 0.1;
      const Vec rim_center = cp.Pos() + ca*kCupRim;
      const double enter = Crossing(lip, velocity, rim_center, ca);
      if (enter > 0.0 && ca.Z() > 0.5 && (lip-rim_center).Dot(ca) > 0.0)
      {
        const Vec offset = StreamPoint(lip, velocity, enter) - rim_center;
        receiving = (offset - ca*offset.Dot(ca)).Length() < kCupRadius-kStreamRadius;
      }
      // Rendering stops on actual Gazebo geometry. This information is used
      // only for water, never published to the teleoperation controller.
      if (ray_)
      {
        Vec previous = lip + ba*0.003;
        for (int i = 1; i <= 24; ++i)
        {
          const double t = duration*i/24.0;
          const Vec next = StreamPoint(lip, velocity, t);
          ray_->SetPoints(previous, next);
          double distance;
          std::string entity;
          ray_->GetIntersection(distance, entity);
          if (!entity.empty() && entity.find("ground_object_left") == std::string::npos
              && distance >= 0 && distance < (next-previous).Length())
          {
            const double hit = duration*(i-1 + distance/(next-previous).Length())/24.0;
            receiving = receiving && hit >= enter-1e-4;
            duration = hit;
            break;
          }
          previous = next;
        }
      }
      const double amount = std::min(bottle_water_, rate*dt);
      bottle_water_ -= amount;
      if (receiving) cup_water_ += amount;
      else spilled_ += amount;
    }
    // A tilted or overfilled cup cannot retain water above its lowest rim.
    const double rim_low = cp.Pos().Z() + ca.Z()*kCupRim
        - kCupRadius*std::sqrt(std::max(0.0, 1-ca.Z()*ca.Z()));
    const double capacity = ca.Z() > 0.0
        ? VolumeBelow(cp, kCupRadius, kCupFloor, kCupRim, rim_low) : 0.0;
    if (cup_water_ > capacity)
    {
      spilled_ += cup_water_-capacity;
      cup_water_ = capacity;
    }
    const double level = cup_water_/(kPi*kCupRadius*kCupRadius);
    if (receiving)
    {
      const Vec surface = cp.Pos() + ca*(kCupFloor+level);
      const double water_hit = Crossing(lip, velocity, surface, ca);
      if (water_hit > 0.0) duration = std::min(duration, water_hit);
    }
    for (int i = 0; i < kSegments; ++i)
    {
      const Vec a = StreamPoint(lip, velocity, duration*i/kSegments);
      const Vec b = StreamPoint(lip, velocity, duration*(i+1)/kSegments);
      const Vec d = b-a;
      ignition::math::Quaterniond q;
      if (d.Length() > 1e-8)
        q.From2Axes(Vec::UnitZ, d.Normalized());
      DrawCylinder("stream_" + std::to_string(i), Pose((a+b)*0.5, q),
                   kStreamRadius, d.Length()+0.0005, rate > 1e-9, 0.18);
    }
    DrawCylinder("cup_fill", cp * Pose(0,0,kCupFloor+level*0.5,0,0,0),
                 kCupRadius-0.0004, level, cup_water_ > 1e-9, 0.28);
    const double bottle_level = bottle_water_/(kPi*kBottleRadius*kBottleRadius);
    DrawCylinder("bottle_fill", bp * Pose(0,0,kBottleBottom+bottle_level*0.5,0,0,0),
                 kBottleRadius, bottle_level, bottle_water_ > 1e-9, 0.48);
    std_msgs::String state;
    std::ostringstream json;
    json << std::fixed << std::setprecision(5)
         << "{\"model\":\"simplified_visual_pouring\",\"sim_time_s\":" << now
         << ",\"bottle_ml\":" << bottle_water_*1e6
         << ",\"cup_ml\":" << cup_water_*1e6
         << ",\"spilled_ml\":" << spilled_*1e6
         << ",\"flow_ml_s\":" << rate*1e6
         << ",\"receiving\":" << (receiving ? "true" : "false")
         << ",\"bottle_tilt_deg\":" << std::acos(std::max(-1.0,std::min(1.0,ba.Z())))*180/kPi
         << "}";
    state.data = json.str();
    state_pub_.publish(state);
  }

  physics::WorldPtr world_;
  physics::RayShapePtr ray_;
  event::ConnectionPtr update_;
  transport::NodePtr transport_node_;
  transport::PublisherPtr visual_pub_;
  std::unique_ptr<ros::NodeHandle> ros_node_;
  ros::CallbackQueue queue_;
  ros::Subscriber reset_sub_;
  ros::Publisher state_pub_;
  double initial_ = 94e-6, bottle_water_ = 94e-6;
  double cup_water_ = 0.0, spilled_ = 0.0, last_update_ = -1.0;
  bool reset_requested_ = false;
};
GZ_REGISTER_WORLD_PLUGIN(PouringVisualizationPlugin)
}  // namespace gazebo
