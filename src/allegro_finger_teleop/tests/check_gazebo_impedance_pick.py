"""Observe a cold --impedance-grasp --replay, including lift, support loss and release."""
import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np
import rospy
from gazebo_msgs.msg import ContactsState, ModelStates
from gazebo_msgs.srv import GetLinkState
from scipy.spatial.transform import Rotation
from std_msgs.msg import String

PACKAGE = Path(__file__).resolve().parents[1]


def xyz(pose):
    return np.array([pose.position.x, pose.position.y, pose.position.z])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rospy.init_node("check_impedance_pick", anonymous=True)
    with gzip.open(PACKAGE / "data/impedance_pick_camera.jsonl.gz", "rt") as f:
        examples = [json.loads(line) for line in f]
    stage = {row["packet"]["sequence"]: row["stage"] for row in examples}
    latest = {"stage": "", "object": None, "contacts": frozenset(), "contact_time": -np.inf, "mode": ""}
    def camera(msg):
        packet = json.loads(msg.data)
        if packet.get("session_id") == "allegro-synthetic-index-pick":
            latest["stage"] = stage.get(packet["sequence"], "")
    def model(msg):
        if "target_object" in msg.name:
            latest["object"] = msg.pose[msg.name.index("target_object")]
    def contacts(msg):
        names = " ".join(s.collision1_name+" "+s.collision2_name for s in msg.states)
        latest["contacts"] = frozenset(k for k in ("pinch_stand", "ground_plane", "algr_lh_if_ds_tip", "algr_lh_th_ds_tip") if k in names)
        latest["contact_time"] = msg.header.stamp.to_sec()
    subscriptions = [
        rospy.Subscriber("/fr3_allegro/camera_packet", String, camera, queue_size=4),
        rospy.Subscriber("/gazebo/model_states", ModelStates, model, queue_size=1),
        rospy.Subscriber("/handarm_sim_demo/target_contacts", ContactsState, contacts, queue_size=4),
        rospy.Subscriber("/fr3_allegro/object_pinch", String, lambda m: latest.update(mode=m.data), queue_size=2)]
    rospy.wait_for_service("/gazebo/get_link_state", timeout=40.)
    link = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
    samples = []
    deadline = time.monotonic()+100
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        if rospy.get_param("/fr3_allegro/replay_complete", False):
            break
        pose = latest["object"]
        if latest["stage"] and pose is not None:
            palm = link("fr3_allegro::algr_lh_palm", "world")
            if not palm.success:
                raise RuntimeError(palm.status_message)
            p = palm.link_state.pose
            q = p.orientation
            rotation = Rotation.from_quat([q.x, q.y, q.z, q.w])
            relative = rotation.inv().apply(xyz(pose)-xyz(p))
            now = rospy.get_time()
            touch = latest["contacts"] if -.03 <= now-latest["contact_time"] < .10 else frozenset()
            samples.append((latest["stage"], now, xyz(pose), relative, touch, latest["mode"]))
        time.sleep(.025)
    for sub in subscriptions:
        sub.unregister()
    groups = {s: [v for v in samples if v[0] == s] for s in ("open", "hold", "lower", "release")}
    if min(map(len, groups.values())) < 12:
        raise RuntimeError("missing an entire demonstration phase")
    hold = [r for r in groups["hold"] if r[1] >= groups["hold"][0][1]+2.]
    if len(hold) < 12:
        raise RuntimeError("hold phase ended before it could be measured")
    rest = np.median([r[2][2] for r in groups["open"]], axis=0)
    before_release = groups["lower"][-12:]
    released = groups["release"][-12:]
    end = np.median([r[2][2] for r in released])
    relative = np.array([r[3] for r in hold])
    heights = np.array([r[2][2] for r in hold])
    pads = {"algr_lh_if_ds_tip", "algr_lh_th_ds_tip"}
    supports = {"pinch_stand", "ground_plane"}
    result = {
        "settled_hold_sim_s": hold[-1][1]-hold[0][1],
        "minimum_lift_mm": 1000*float(heights.min()-rest),
        "maximum_palm_relative_drift_mm": 1000*float(np.linalg.norm(relative-relative[0], axis=1).max()),
        "bilateral_pad_fraction": float(np.mean([pads.issubset(r[4]) for r in hold])),
        "supported_samples_during_hold": sum(bool(r[4] & supports) for r in hold),
        "spring_preload_fraction": float(np.mean([r[5]=="index" for r in hold])),
        "lowered_on_two_pads_fraction": float(np.mean([pads.issubset(r[4]) and not (r[4] & supports) for r in before_release])),
        "lower_end_contacts_fraction": {name: float(np.mean([name in r[4] for r in before_release]))
                                        for name in sorted(pads | supports)},
        "hold_end_world_mm": np.round(1000*hold[-1][2], 2).tolist(),
        "lower_end_world_mm": np.round(1000*before_release[-1][2], 2).tolist(),
        "release_end_world_mm": np.round(1000*released[-1][2], 2).tolist(),
        "release_drop_mm": 1000*float(np.median([r[2][2] for r in before_release])-end),
        "stand_after_release_fraction": float(np.mean(["pinch_stand" in r[4] and not (r[4] & pads) for r in released])),
        "released_height_error_mm": 1000*abs(float(end-rest)),
        "hold_samples": len(hold),
    }
    result["passed"] = bool(result["settled_hold_sim_s"] >= 3.5 and result["minimum_lift_mm"] >= 45.
        and result["maximum_palm_relative_drift_mm"] <= 6. and result["bilateral_pad_fraction"] >= .9
        and result["supported_samples_during_hold"] == 0 and result["spring_preload_fraction"] >= .9
        and result["lowered_on_two_pads_fraction"] >= .9 and result["release_drop_mm"] >= 5.
        and result["stand_after_release_fraction"] >= .9 and result["released_height_error_mm"] <= 5.)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
