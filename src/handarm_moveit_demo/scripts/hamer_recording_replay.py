#!/usr/bin/env python3
"""Replay an existing perception_hamer JSONL session with aligned D455 depth."""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import rospy

from handarm_moveit_demo.msg import HamerHandPose
from handarm_moveit_demo.shared_teleop_core import matrix_to_quaternion_xyzw


WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
from perception_hamer.src.teleop_pose_packet import metric_wrist_from_arrays  # noqa: E402
from perception_hamer.src.finger_observation import observe_mano_fingers  # noqa: E402


class HamerRecordingReplay:
    def __init__(self):
        config = rospy.get_param("/shared_teleop", {})
        topics = config.get("topics", {})
        self.session = Path(rospy.get_param("~session_directory")).expanduser().resolve()
        self.speed = float(rospy.get_param("~speed", 1.0))
        if self.speed <= 0.0 or not (self.session/"frames.jsonl").is_file():
            raise ValueError("session_directory must contain frames.jsonl and speed must be positive")
        summary = json.loads((self.session/"summary.json").read_text(encoding="utf-8"))
        device = summary.get("device", {})
        self.intrinsics = device.get("color_intrinsics")
        self.depth_scale = device.get("depth_scale_m_per_unit")
        if self.intrinsics is None or self.depth_scale is None:
            raise ValueError("HaMeR session lacks D455 color intrinsics/depth scale")
        self.records = [json.loads(line) for line in
                        (self.session/"frames.jsonl").read_text(encoding="utf-8").splitlines()
                        if line.strip()]
        self.records = [record for record in self.records if record.get("valid")]
        if not self.records:
            raise ValueError("HaMeR session has no valid frames")
        self.publisher = rospy.Publisher(topics.get("hamer_pose", "/shared_teleop/hamer_pose"),
                                         HamerHandPose, queue_size=1)

    def safe_depth_path(self, record):
        path = (self.session/record["aligned_depth_path"]).resolve()
        if self.session not in path.parents:
            raise ValueError("recorded depth path escapes session")
        return path

    def message(self, record, sequence):
        joint = record.get("palm_frames", {}).get("mano_joint_palm_frame", {})
        if not joint.get("valid"):
            raise ValueError("record has no valid MANO joint palm frame")
        depth = cv2.imread(str(self.safe_depth_path(record)), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError("failed to load aligned depth")
        position, depth_confidence, _ = metric_wrist_from_arrays(
            record["mano_joints_2d_crop_normalized"],
            record["hamer_quality"]["affine_original_to_crop"],
            depth, self.depth_scale, self.intrinsics)
        quaternion = (np.asarray(joint["quaternion_xyzw"], dtype=float)
                      if joint.get("quaternion_xyzw") is not None else
                      matrix_to_quaternion_xyzw(joint["rotation"]))
        roi_confidence = float(np.clip(record.get("roi", {}).get("confidence", 0.0), 0.0, 1.0))
        rotation_confidence = roi_confidence*float(np.clip(
            record.get("hamer_quality", {}).get("bbox_visible_fraction", 0.0), 0.0, 1.0))
        finger_observation = observe_mano_fingers(
            record.get("mano_joints"),
            roi_confidence,
            record.get("hamer_quality", {}).get("bbox_visible_fraction", 0.0),
            record.get("crop_quality", 1.0),
        )
        message = HamerHandPose()
        message.header.seq = sequence
        message.header.stamp = rospy.Time.now()
        message.source_timestamp = float(record["timestamp"])
        message.header.frame_id = "camera_color_optical_frame"
        message.wrist_pose.position.x, message.wrist_pose.position.y, message.wrist_pose.position.z = position
        (message.wrist_pose.orientation.x, message.wrist_pose.orientation.y,
         message.wrist_pose.orientation.z, message.wrist_pose.orientation.w) = quaternion
        message.confidence = [rotation_confidence*depth_confidence]*3+[rotation_confidence]*3
        message.finger_tracking_present = True
        message.finger_tracking_valid = finger_observation.valid
        message.finger_flexion = finger_observation.flexion.tolist()
        message.finger_tracking_confidence = finger_observation.confidence
        message.finger_invalid_reason = finger_observation.invalid_reason
        message.valid = True
        # Existing recordings contain hand_pose but no validated discrete gesture classifier.
        message.gesture = int(record.get("gesture", 0))
        message.gesture_confidence = float(record.get("gesture_confidence", 0.0))
        message.invalid_reason = ""
        return message

    def run(self):
        first_stamp = float(self.records[0]["timestamp"])
        started = rospy.get_time()
        published = 0
        skipped = {}
        for record in self.records:
            due = started+(float(record["timestamp"])-first_stamp)/self.speed
            while not rospy.is_shutdown() and rospy.get_time() < due:
                rospy.sleep(min(0.01, due-rospy.get_time()))
            if rospy.is_shutdown():
                break
            try:
                self.publisher.publish(self.message(record, published))
                published += 1
            except Exception as exc:
                reason = str(exc)
                skipped[reason] = skipped.get(reason, 0)+1
                rospy.logwarn_throttle(1.0, "Recorded HaMeR frame skipped: %s", reason)
        rospy.loginfo(
            "HaMeR recording replay complete: %d metric wrist poses; skipped=%s",
            published, json.dumps(skipped, sort_keys=True, separators=(",", ":")))


def main():
    rospy.init_node("hamer_recording_replay")
    HamerRecordingReplay().run()


if __name__ == "__main__":
    main()
