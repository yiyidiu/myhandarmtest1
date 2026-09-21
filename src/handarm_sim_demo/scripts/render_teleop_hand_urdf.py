#!/usr/bin/env python3
"""Render reversible hand-control variants of the Gazebo URDF.

The canonical ``gazebo_handarm.urdf`` is deliberately left unchanged.  During
arm teleoperation no node publishes a hand trajectory, so fighting moving-base
inertia with the very light finger links and finite-effort PIDs only creates a
visible oscillation.  ``rigid_transport`` removes the mimic PID selector; the
four active joints likewise use gazebo_ros_control's native position path when
their plant PIDs are not loaded by the launch file.  Hand trajectories still
work, but this mode is kinematic and is not the physical-contact validation
profile.  ``physical_grasp`` uses self-locking worm drives only at the four
actuated joints, with passive spatial transmission constraints for 1:1 followers.
Contact loads are transmitted through the coupled finger to its root motor.
The physical profile also
uses a bounded-position arm backend that handles joint_6's multi-turn range
without changing its URDF limits or PID gains. ``--profile original``
returns the input byte-for-byte.
"""

import argparse
import copy
import sys
import xml.etree.ElementTree as ET


ACTIVE_JOINTS = ("f1j1", "f1j2", "f2j1", "f3j2")
MIMIC_JOINTS = ("f3j1", "f1j3", "f2j2", "f3j3")
ALL_HAND_JOINTS = ACTIVE_JOINTS + MIMIC_JOINTS
FINGER_LINKS = (
    "f1link1", "f1link2", "f1link3",
    "f2link1", "f2link2",
    "f3link1", "f3link2", "f3link3",
)


def render_rigid_transport_hand(xml_text):
    """Return a URDF whose mimic joints use Gazebo's position path."""

    root = ET.fromstring(xml_text)
    found_mimics = set()
    for plugin in root.findall("./gazebo/plugin"):
        mimic_element = plugin.find("mimicJoint")
        if mimic_element is None or mimic_element.text not in MIMIC_JOINTS:
            continue
        pid_selector = plugin.find("hasPID")
        if pid_selector is not None:
            plugin.remove(pid_selector)
        # A SetVelocityLimit constraint on a force/position-coupled joint
        # injects solver impulses in Gazebo 11. Never retain one here.
        velocity_limit = plugin.find("maxVelocity")
        if velocity_limit is not None:
            plugin.remove(velocity_limit)
        found_mimics.add(mimic_element.text)

    missing_mimics = sorted(set(MIMIC_JOINTS) - found_mimics)
    if missing_mimics:
        raise ValueError(
            "input URDF is missing mimic plugins for: {}".format(
                ", ".join(missing_mimics)))

    return ET.tostring(root, encoding="unicode")


def _space_join(values):
    return " ".join(str(value) for value in values)


def _add_text(parent, tag, text):
    element = ET.SubElement(parent, tag)
    element.text = str(text)
    return element


def _set_text(parent, tag, text):
    element = parent.find(tag)
    if element is None:
        element = ET.SubElement(parent, tag)
    element.text = str(text)
    return element


def render_physical_grasp_hand(xml_text):
    """Return a contact-capable self-locking hand plant.

    The arm remains owned by gazebo_ros_control. The eight hand joints are
    deliberately removed from that hardware interface and are all owned by a
    single Gazebo plugin. Only active joints receive motor references;
    follower joints use passive fixed-ratio constraints rather than servos.
    """

    root = ET.fromstring(xml_text)
    for plugin in root.findall("gazebo/plugin"):
        if plugin.get("filename") == "libgazebo_ros_control.so":
            _set_text(plugin, "robotSimType", "handarm_sim_demo/BoundedPositionHWSim")

    removed_transmissions = set()
    for transmission in list(root.findall("transmission")):
        joint = transmission.find("joint")
        if joint is None or joint.get("name") not in ACTIVE_JOINTS:
            continue
        removed_transmissions.add(joint.get("name"))
        root.remove(transmission)
    missing_transmissions = sorted(
        set(ACTIVE_JOINTS) - removed_transmissions)
    if missing_transmissions:
        raise ValueError(
            "input URDF is missing active hand transmissions for: {}".format(
                ", ".join(missing_transmissions)))

    removed_mimics = set()
    for gazebo in list(root.findall("gazebo")):
        for plugin in list(gazebo.findall("plugin")):
            mimic = plugin.findtext("mimicJoint")
            if mimic not in MIMIC_JOINTS:
                continue
            removed_mimics.add(mimic)
            gazebo.remove(plugin)
        if not list(gazebo) and not gazebo.attrib:
            root.remove(gazebo)
    missing_mimics = sorted(set(MIMIC_JOINTS) - removed_mimics)
    if missing_mimics:
        raise ValueError(
            "input URDF is missing mimic plugins for: {}".format(
                ", ".join(missing_mimics)))

    # The decimated wrist meshes bridge empty space: at the recorded stop
    # link_4/link_6 report 1.05 mm while their displayed surfaces are 16.87 mm
    # apart. These modest meshes (about 6300/900 faces) retain the real outer
    # shape in both ODE and MoveIt without changing visuals or inertias.
    # The old link_5 proxy also bridges empty space at the palm mount.
    # Sep 19: the lower-arm coarse hull also fills a recess near finger 3.
    # At the recorded 1:33 stop it reports 6.45 mm, while the CAD surfaces
    # are 12.03 mm apart. Use the modest 2888/5740-face lower-arm meshes too;
    # retain clearance/braking margins and all collision pairs.
    # The palm's rounded 2 mm mesh also bridges empty space. At the Sep 19
    # fault it intersects link_4 while the original CAD retains 1.197 mm.
    # Use the original palm for BOTH simulation contact and every MoveIt
    # distance/collision query; an exact binary check alone would leave the
    # QP and trajectory sweep rejecting the same false contact.
    for name in ("link_1", "link_2", "link_4", "link_5", "link_6", "handbase_link"):
        link = root.find("link[@name='{}']".format(name))
        visual = link.find("visual")
        for collision in list(link.findall("collision")):
            link.remove(collision)
        collision = ET.SubElement(link, "collision")
        for key in ("origin", "geometry"):
            element = visual.find(key)
            if element is not None:
                collision.append(copy.deepcopy(element))

    # Replace the root bounding boxes that created an invisible bottle
    # support. Low-face convex hulls retain the outer root shape without
    # putting the 160k-face visual meshes in the real-time collision loop.
    # Finger capsules falsely overlapped the palm at f3j2=1.23865 rad while
    # the original finger/palm surfaces were 6.35 mm apart. A single convex
    # hull also fills the proximal link's side recess, so preserve it with
    # three longitudinal convex sections. The sections contain the original
    # CAD surfaces; no shrinking, joint-limit reduction or collision-pair
    # exemption is used. See meshes/finger_collision_geometry.md.
    for name in FINGER_LINKS:
        link = root.find("link[@name='{}']".format(name))
        visual = link.find("visual")
        for collision in list(link.findall("collision")):
            link.remove(collision)
        part_count = 3 if name in ("f1link2", "f2link1", "f3link2") else 1
        for part in range(part_count):
            collision = ET.SubElement(link, "collision")
            for key in ("origin", "geometry"):
                element = visual.find(key)
                if element is not None:
                    collision.append(copy.deepcopy(element))
            suffix = "hull" if name in ("f1link1", "f3link1") else "part{}".format(part)
            collision.find("geometry/mesh").set(
                "filename", "package://handarm_sim_demo/meshes/{}_collision_{}.stl".format(name, suffix))

    # Gazebo's default contact stiffness is much too hard for these small
    # finger inertias. A moderately compliant, damped surface avoids the
    # stick/bounce limit cycle while retaining the established high friction.
    configured_links = set()
    for gazebo in root.findall("gazebo"):
        link_name = gazebo.get("reference")
        if link_name not in FINGER_LINKS:
            continue
        _set_text(gazebo, "kp", "100000.0")
        _set_text(gazebo, "kd", "100.0")
        _set_text(gazebo, "maxVel", "0.02")
        _set_text(gazebo, "minDepth", "0.001")
        configured_links.add(link_name)
    missing_links = sorted(set(FINGER_LINKS) - configured_links)
    if missing_links:
        raise ValueError(
            "input URDF is missing Gazebo finger surfaces for: {}".format(
                ", ".join(missing_links)))

    # Passive/arm damping remains implicit. Active worm-drive brakes own
    # their ODE stops, with motion bounded by the original motor speed.
    for joint_name in ALL_HAND_JOINTS + tuple(
            "joint_{}".format(index) for index in range(1, 7)):
        joint_gazebo = ET.SubElement(root, "gazebo", {"reference": joint_name})
        _add_text(joint_gazebo, "implicitSpringDamper",
                  "false" if joint_name in ACTIVE_JOINTS else "true")

    wrapper = ET.SubElement(root, "gazebo")
    plugin = ET.SubElement(
        wrapper, "plugin",
        {"name": "stable_physical_grasp_hand",
         "filename": "libhandarm_stable_hand_spring_plugin.so"})
    _add_text(plugin, "robotNamespace", "/")
    _add_text(plugin, "activeJointNames", _space_join(ACTIVE_JOINTS))
    _add_text(plugin, "mimicJointNames", _space_join(MIMIC_JOINTS))
    _add_text(plugin, "mimicSourceIndex", "0 1 2 3")
    _add_text(plugin, "initialPositions", "0.18 0.08726646259971647 0.08726646259971647 0.08726646259971647")
    # Position-mode motor gains/effort only; non-backdrivability comes from
    # mechanical shaft constraints, not a compliant holding spring.
    _add_text(plugin, "activeStiffness", "25.0 12.0 12.0 12.0")
    # Only the four actuated joints have a drive. The three distal joints
    # and mirrored spread joint use passive 1:1 transmission constraints.
    # These are joint torque caps, not a calibrated fingertip force in N.
    _add_text(plugin, "activeMaxEffort", "3.0 1.80 1.80 1.80")
    _add_text(plugin, "commandTopic", "/controller_gazebo_hand/command")
    _add_text(plugin, "stateTopic", "/joint_states")
    _add_text(
        plugin, "diagnosticTopic",
        "/handarm_sim_demo/physical_hand_diagnostics")
    _add_text(plugin, "publishRate", "50.0")

    return ET.tostring(root, encoding="unicode")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--profile",
        choices=("original", "rigid_transport", "physical_grasp"),
        default="physical_grasp")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(args.input, "r", encoding="utf-8") as stream:
        xml_text = stream.read()
    if args.profile == "original":
        sys.stdout.write(xml_text)
        return 0
    if args.profile == "rigid_transport":
        rendered = render_rigid_transport_hand(xml_text)
    else:
        rendered = render_physical_grasp_hand(xml_text)
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
