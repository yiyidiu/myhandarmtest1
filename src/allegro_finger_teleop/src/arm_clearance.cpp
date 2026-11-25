// Collision geometry for the arm reference. Finger-to-finger contacts belong to
// the grasp task. Measured finger angles still move all hand shapes in this check.
#include <fcl/fcl.h>
#include <urdf_parser/urdf_parser.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Transform = fcl::Transform3d;
using Vector = fcl::Vector3d;
Transform pose(const urdf::Pose& p) {
  Transform t = Transform::Identity();
  t.linear() =
      Eigen::Quaterniond(p.rotation.w, p.rotation.x, p.rotation.y, p.rotation.z).toRotationMatrix();
  t.translation() = Vector(p.position.x, p.position.y, p.position.z);
  return t;
}
struct Body {
  int parent, joint;
  Transform origin;
  Vector axis;
  std::string name;
};
struct Shape {
  int body;
  Transform origin;
  std::shared_ptr<fcl::CollisionGeometryd> geometry;
  double radius;
};
struct Guard {
  std::vector<std::string> names;
  std::vector<Body> bodies;
  std::vector<Shape> shapes;
  using Pairs = std::vector<std::pair<int, int>>;
  Pairs pairs, thumb_housings;
  std::vector<Transform> links, transforms;
  explicit Guard(const char* xml, const char* ordered_names) {
    std::istringstream in(ordered_names);
    for (std::string s; in >> s;) names.push_back(s);
    auto model = urdf::parseURDF(xml);
    if (!model || names.size() != 23) throw std::runtime_error("invalid robot");
    visit(model->getRoot(), -1);
    auto is_hand = [&](int i) { return bodies[i].name.find("algr_lh_") == 0; };
    for (size_t i = 0; i < shapes.size(); ++i)
      for (size_t j = i + 1; j < shapes.size(); ++j) {
        int a = shapes[i].body, b = shapes[j].body;
        const auto housing_pair = [](const std::string& thumb, const std::string& housing) {
          return (thumb == "algr_lh_th_px" || thumb == "algr_lh_th_ds") &&
              (housing == "algr_lh_palm" || housing == "algr_lh_if_bs" ||
               housing == "algr_lh_mf_bs" || housing == "algr_lh_rf_bs");
        };
        if (housing_pair(bodies[a].name, bodies[b].name) || housing_pair(bodies[b].name, bodies[a].name))
          thumb_housings.emplace_back(i, j);
        if (a == b || bodies[a].parent == b || bodies[b].parent == a || (is_hand(a) && is_hand(b)))
          continue;
        pairs.emplace_back(i, j);
      }
    links.resize(bodies.size());
    transforms.resize(shapes.size());
  }
  void addTableBox(const double* transform, const double* size) {
    Transform t = Transform::Identity();
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 4; j++) t.matrix()(i, j) = transform[4 * i + j];
    Vector dimensions(size[0], size[1], size[2]);
    int body = bodies.size(), index = shapes.size();
    bodies.push_back({-1, -1, t, Vector::UnitZ(), "workbench"});
    shapes.push_back(
        {body, Transform::Identity(), std::make_shared<fcl::Boxd>(dimensions), .5 * dimensions.norm()});
    for (int i = 0; i < index; i++) {
      const auto& name = bodies[shapes[i].body].name;
      if (name != "fr3_link0" && name != "workbench") pairs.emplace_back(i, index);
    }
    links.resize(bodies.size());
    transforms.resize(shapes.size());
  }
  void visit(const urdf::LinkConstSharedPtr& l, int parent) {
    int n = bodies.size(), joint = -1;
    Vector axis = Vector::UnitZ();
    Transform t = Transform::Identity();
    if (l->parent_joint) {
      auto j = l->parent_joint;
      t = pose(j->parent_to_joint_origin_transform);
      if (j->type != urdf::Joint::FIXED) {
        auto it = std::find(names.begin(), names.end(), j->name);
        if (it == names.end()) throw std::runtime_error("unknown joint");
        joint = it - names.begin();
        axis = Vector(j->axis.x, j->axis.y, j->axis.z).normalized();
      }
    }
    bodies.push_back({parent, joint, t, axis, l->name});
    for (const auto& c : l->collision_array) {
      auto g = c->geometry;
      std::shared_ptr<fcl::CollisionGeometryd> geom;
      double r;
      if (g->type == urdf::Geometry::SPHERE) {
        auto s = static_cast<const urdf::Sphere*>(g.get());
        r = s->radius;
        geom = std::make_shared<fcl::Sphered>(r);
      } else if (g->type == urdf::Geometry::BOX) {
        auto b = static_cast<const urdf::Box*>(g.get());
        Vector d(b->dim.x, b->dim.y, b->dim.z);
        r = .5 * d.norm();
        geom = std::make_shared<fcl::Boxd>(d);
      } else if (g->type == urdf::Geometry::CYLINDER) {
        auto v = static_cast<const urdf::Cylinder*>(g.get());
        r = std::hypot(v->radius, .5 * v->length);
        geom = std::make_shared<fcl::Cylinderd>(v->radius, v->length);
      } else
        throw std::runtime_error("unsupported collision shape");
      shapes.push_back({n, pose(c->origin), geom, r});
    }
    for (const auto& ch : l->child_links) visit(ch, n);
  }
  // Positive clearance, or signed penetration. Clamp far-away pairs to avoid
  // running narrow-phase distance on every combination at each reference tick.
  void distances(const double* q, const Pairs& selected, std::vector<double>& out) {
    for (size_t i = 0; i < bodies.size(); ++i) {
      const auto& b = bodies[i];
      links[i] = b.parent < 0 ? b.origin : links[b.parent] * b.origin;
      if (b.joint >= 0) links[i].linear() *= Eigen::AngleAxisd(q[b.joint], b.axis).toRotationMatrix();
    }
    for (size_t i = 0; i < shapes.size(); ++i) transforms[i] = links[shapes[i].body] * shapes[i].origin;
    out.resize(selected.size());
    for (size_t k = 0; k < selected.size(); ++k) {
      int i = selected[k].first, j = selected[k].second;
      const auto& a = shapes[i];
      const auto& b = shapes[j];
      if ((transforms[i].translation() - transforms[j].translation()).norm() - a.radius - b.radius >
          .012) {
        out[k] = .012;
        continue;
      }
      fcl::DistanceRequestd request(false, true);
      fcl::DistanceResultd result;
      out[k] = std::min(.012, fcl::distance(a.geometry.get(), transforms[i], b.geometry.get(),
                                            transforms[j], request, result));
    }
  }
  bool accepts(const double* start, const double* finish, const Pairs& selected, double margin) {
    std::vector<double> previous, next, q(names.size());
    distances(start, selected, previous);
    double longest = 0;
    for (size_t j = 0; j < names.size(); ++j)
      longest = std::max(longest, std::abs(start[j] - finish[j]));
    int count = std::max(1, static_cast<int>(std::ceil(longest / .012)));
    for (int i = 1; i <= count; ++i) {
      for (size_t j = 0; j < names.size(); ++j) q[j] = start[j] + (finish[j] - start[j]) * i / count;
      distances(q.data(), selected, next);
      for (size_t k = 0; k < next.size(); ++k)
        if (!std::isfinite(next[k]) || next[k] + 1e-8 < std::min(margin, previous[k])) return false;
      previous.swap(next);
    }
    return true;
  }
};
}  // namespace
extern "C" void* fr3_guard_new(const char* xml, const char* names) {
  try {
    return new Guard(xml, names);
  } catch (...) {
    return nullptr;
  }
}
extern "C" void fr3_guard_free(void* handle) { delete static_cast<Guard*>(handle); }
extern "C" int fr3_guard_allows(void* handle, const double* a, const double* b) {
  try {
    auto* guard = static_cast<Guard*>(handle);
    return guard->accepts(a, b, guard->pairs, .004);
  } catch (...) {
    return 0;
  }
}
extern "C" int fr3_guard_table(void* handle, const double* pose, const double* size) {
  try {
    static_cast<Guard*>(handle)->addTableBox(pose, size);
    return 1;
  } catch (...) {
    return 0;
  }
}

extern "C" int fr3_thumb_housings_allow(void* handle, const double* a, const double* b) {
  try {
    auto* guard = static_cast<Guard*>(handle);
    return !guard->thumb_housings.empty() && guard->accepts(a, b, guard->thumb_housings, .003);
  } catch (...) {
    return 0;
  }
}
