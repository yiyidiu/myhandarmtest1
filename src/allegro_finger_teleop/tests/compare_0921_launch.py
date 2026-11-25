"""Compare resolved live ROS settings with the top-level September 21 release."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import os
import sys
from xml.etree import ElementTree as ET
import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[3]
TAG = "teleop-whole-wrist-scenes-20260921"
SCOPES = ("six_dof_trend","moveit_servo_output_adapter","hamer_input_adapter","shared_teleop",
          "full_robot_self_collision_guard","egm_position_reference_adapter")
# Every permitted difference concerns a concrete model, hand or transport interface.
HARDWARE = {
    "hamer_input_adapter": {"port","packet_topic","articulation_hold_topic","required_wrist_baseline","required_position_source"},
    "shared_teleop": {"assistance/move_group","control/articulation_hold_topic","control/arm_joints","control/operational_joint_limits_deg",
                      "control/operational_joint_limits_deg/joint_3","control/operational_joint_limits_deg/joint_6",
                      "control/target_ik_group","workspace/minimum_base_m","workspace/maximum_base_m"},
    "full_robot_self_collision_guard": {"simulated_arm_stop_deflection_rad","redundant_joint_centering_rad_s","joint_motion_weights","reserve_hardware_stopping_distance","verify_self_detour_constraint","use_abb_visual_palm","joint_state_topic","grasp_contact_links",
        "arm_group","arm_joint_names","hand_command_topic","hand_reference_joint_names","initial_hand_reference","required_joint_names"},
    "egm_position_reference_adapter": {"arm_joints","controller_topic","initial_reference","initial_reference_param",
        "joint_limit_margin_rad","lower_limits","upper_limits","maximum_acceleration","maximum_following_error_rad","maximum_velocity"},
}
def resolved(source, original):
    if original:
        entry = 'source "$1/devel/setup.bash" --extend; shift; exec roslaunch --dump-params handarm_moveit_demo live_human_ground_gazebo_egm_teleop.launch "$@"'
        args = ["gazebo_gui:=false","grasp_alignment_views:=false","input_source:=udp","mapping_profile:=whole_wrist_pose",
                "constrained_velocity_projection:=true","joint_velocity_solver:=qp","hamer_maximum_pipeline_latency_s:=0.25",
                "enable_logger:=false","ground_object_world:="+str(source/"src/handarm_sim_demo/worlds/handarm_pouring.world")]
    else:
        entry = 'source "$1/.build/robot/devel/setup.bash" --extend; source "$1/.build/core/devel/setup.bash" --extend; export ROS_PACKAGE_PATH="$1/src:$1/reproduction/fr3:$ROS_PACKAGE_PATH"; shift; exec roslaunch --dump-params allegro_finger_teleop fingers.launch "$@"'
        args = ["gui:=false","source:=external"]
    env = {k:v for k,v in os.environ.items() if k in ("HOME","USER","PATH","LANG","LC_ALL")}
    env.update(TELEOP_CONTROL_REFERENCE="mano-wrist-ring", TELEOP_SELF_DETOUR="true")
    raw = subprocess.check_output(["bash","-c","set -e; source /opt/ros/noetic/setup.bash --local; "+entry,
                                   "audit",str(source),*args],env=env)
    values = yaml.safe_load(raw)
    # roslaunch uploads the explicit final empty dictionary after ABB's child keys.
    for key in tuple(values):
        if any(values.get(key[:i])=={} for i,char in enumerate(key) if char=="/"):
            del values[key]
    return values

def check_core_sources(original):
    paths = (
        "perception_hamer/src/causal_wrist_so3_filter.py",
        "src/handarm_moveit_demo/src/handarm_moveit_demo/shared_teleop_core.py",
        "src/handarm_moveit_demo/src/handarm_moveit_demo/egm_position_reference.py",
        "src/handarm_moveit_demo/scripts/moveit_servo_output_adapter.py",
        "src/handarm_moveit_demo/scripts/shared_control_node.py",
        "src/handarm_moveit_demo/scripts/gesture_isolation_node.py",
    )
    hashes = {}
    for path in paths:
        content = (PROJECT / path).read_bytes()
        assert content == (original / path).read_bytes(), path
        hashes[path] = hashlib.sha256(content).hexdigest()
    return hashes


def configured_mapper(root, params, module_name):
    path = root / "src/handarm_moveit_demo/src/handarm_moveit_demo/shared_teleop_core.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    core = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = core
    spec.loader.exec_module(core)
    prefix = "/shared_teleop/mapping_profiles/whole_wrist_pose/"
    side = params["/shared_teleop/mapping/maximum_side_rotation_deg"]
    margin = params["/shared_teleop/mapping/side_rotation_tracking_margin_deg"]
    mapper = core.WholeWristPoseMapper(
        translation_matrix=params[prefix + "translation_matrix"],
        translation_gain=params[prefix + "translation_gain"],
        orientation_gain=params[prefix + "orientation_gain"],
        maximum_relative_translation=params["/shared_teleop/control/maximum_relative_translation_m"],
        maximum_relative_rotation_rad=np.deg2rad(params["/shared_teleop/control/maximum_relative_rotation_deg"]),
        maximum_side_rotation_rad=np.deg2rad(side - margin),
    )
    mapper.rotation_matrix = mapper.rotation_matrix @ np.array(params[prefix + "wrist_ring_to_palm_basis"]).T
    return mapper, core.so3_exp


def check_six_dof_targets(original, original_params, fr3_params):
    original_mapper, so3_exp = configured_mapper(original, original_params, "original_pose_math")
    fr3_mapper, _ = configured_mapper(PROJECT, fr3_params, "fr3_pose_math")
    vectors = [(np.zeros(3), np.zeros(3))]
    for axis in np.eye(3):
        for sign in (-1, 1):
            vectors.append((sign * .035 * axis, np.zeros(3)))
            for angle in (5., 40., 88., 102., 140.):
                vectors.append((np.zeros(3), sign * np.deg2rad(angle) * axis))
    rng = np.random.default_rng(921)
    for _ in range(64):
        axis = rng.normal(size=3)
        angle = np.deg2rad(rng.uniform(0., 160.))
        vectors.append((rng.uniform(-.16, .16, size=3), angle * axis / np.linalg.norm(axis)))
    zero_position, zero_rotation = np.array([.34, -.08, .55]), so3_exp(np.array([.35, -.2, .18]))
    for delta, rotation_vector in vectors:
        inputs = (delta, so3_exp(rotation_vector), zero_position, zero_rotation)
        baseline_target = original_mapper.map(*inputs)
        fr3_target = fr3_mapper.map(*inputs)
        for baseline_value, fr3_value in zip(baseline_target, fr3_target):
            np.testing.assert_array_equal(baseline_value, fr3_value)
    return {"cases": len(vectors), "identical_target_positions_and_rotations": True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",type=Path);args=p.parse_args()
    original=Path(os.environ.get("FR3_SEPTEMBER_CHECKOUT", PROJECT.parent/"whole-wrist"))
    rev=lambda *a: subprocess.check_output(["git","-C",str(original),"rev-parse",*a],text=True).strip()
    commit=rev(TAG+"^{commit}")
    if rev("HEAD")!=commit: raise ValueError("the installed upstream worktree is not the frozen tag")
    a,b=resolved(original,True),resolved(PROJECT,False)
    report=dict(upstream_commit=commit, equal={},hardware={},unexpected=[],
                core_source_sha256=check_core_sources(original), six_dof_targets=check_six_dof_targets(original,a,b))
    # The original task scenes and the FR3 model have different native
    # simulation clocks. Verify the chosen FR3 clock and scene identity.
    hardware = ET.fromstring(b["/robot_description"])
    plugin = next(p for p in hardware.iter("plugin")
                  if p.findtext("robotSimType") == "fr3_allegro_sim/BoundedRevoluteHWSim")
    period = float(plugin.findtext("controlPeriod"))
    assert period == .0005
    for scene in ("ground_grasp", "pouring", "front_stacking", "motion_only"):
        rel = "src/handarm_sim_demo/worlds/handarm_" + scene + ".world"
        before, after = ET.parse(original/rel).getroot(), ET.parse(PROJECT/rel).getroot()
        physics = after.find("world/physics")
        assert float(physics.findtext("max_step_size")) == period
        assert float(physics.findtext("real_time_update_rate")) == 1 / period
        for key in ("max_step_size", "real_time_update_rate"):
            physics.find(key).text = before.findtext("world/physics/" + key)
        assert ET.tostring(before) == ET.tostring(after), scene
    report["fr3_simulation_period_seconds"] = period
    for key in ("lower_singularity_threshold", "hard_stop_singularity_threshold", "joint_limit_margin"):
        path = "/servo_server/" + key
        if key == "joint_limit_margin":
            assert a[path] == 0.08 and b[path] == 0.0, path
        else:
            assert a[path] == b[path], path
        report.setdefault("guard_global_inputs", {})[key] = [a[path], b[path]]
    for group in SCOPES:
        prefix="/"+group+"/"
        same=0; exceptions={}
        for k in sorted(set(a)|set(b)):
            if not k.startswith(prefix): continue
            x,y=a.get(k),b.get(k)
            if k.endswith("/camera_workspace_calibration_file"):
                x,y=(hashlib.sha256(Path(v).read_bytes()).hexdigest() for v in (x,y))
            if x==y: same+=1; continue
            name=k[len(prefix):]
            if name not in HARDWARE.get(group,{}): report["unexpected"].append(k)
            exceptions[name]=dict(original=x,fr3=y)
        report["equal"][group]=same
        if exceptions: report["hardware"][group]=exceptions
    assert b["/six_dof_trend/hand_reference_kind"]=="mano-wrist-ring"
    assert b["/shared_teleop/control/operational_joint_limits_deg"]=={}
    assert b["/hamer_input_adapter/required_position_source"]=="DIRECT_CAMERA_PALM_SKIN_DEPTH_NO_MODEL_OFFSET"
    if report["unexpected"]: raise AssertionError(report["unexpected"])
    out=json.dumps(report,ensure_ascii=False,indent=2)+"\n"
    if args.output: args.output.write_text(out)
    else: print(out)

if __name__=="__main__":
    main()
