#!/usr/bin/env python3
"""Replay raw wrist poses from a shared-teleop CSV log into the ROS adapter."""

import csv
from pathlib import Path

import rospy

from handarm_moveit_demo.msg import HamerHandPose


CONFIDENCE_FIELDS = ["confidence_x", "confidence_y", "confidence_z",
                     "confidence_roll", "confidence_pitch", "confidence_yaw"]


def as_bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


class PoseReplay:
    def __init__(self):
        config = rospy.get_param("/shared_teleop", {})
        topics = config.get("topics", {})
        self.path = Path(rospy.get_param("~input_csv")).expanduser().resolve()
        self.speed = float(rospy.get_param("~speed", 1.0))
        self.loop = bool(rospy.get_param("~loop", False))
        if self.speed <= 0.0 or not self.path.is_file():
            raise ValueError("input_csv must exist and speed must be positive")
        with self.path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.rows = []
        previous = None
        for row in rows:
            source = (row.get("raw_hand_source_stamp") or row.get("raw_hand_stamp") or
                      row.get("timestamp_ros"))
            if not source:
                continue
            stamp = float(source)
            if previous is not None and abs(stamp-previous) < 1.0e-9:
                continue
            previous = stamp
            self.rows.append((stamp, row))
        if not self.rows:
            raise ValueError("CSV contains no replayable raw wrist pose")
        self.publisher = rospy.Publisher(topics.get("hamer_pose", "/shared_teleop/hamer_pose"),
                                         HamerHandPose, queue_size=1)

    def run_once(self):
        first_source = self.rows[0][0]
        started = rospy.get_time()
        for sequence, (source, row) in enumerate(self.rows):
            due = started+(source-first_source)/self.speed
            while not rospy.is_shutdown() and rospy.get_time() < due:
                rospy.sleep(min(0.01, due-rospy.get_time()))
            if rospy.is_shutdown():
                return
            message = HamerHandPose()
            message.header.seq = sequence
            message.header.stamp = rospy.Time.now()
            message.source_timestamp = source
            message.header.frame_id = row.get("raw_hand_frame") or "camera_color_optical_frame"
            message.wrist_pose.position.x = float(row["raw_hand_x"])
            message.wrist_pose.position.y = float(row["raw_hand_y"])
            message.wrist_pose.position.z = float(row["raw_hand_z"])
            message.wrist_pose.orientation.x = float(row["raw_hand_qx"])
            message.wrist_pose.orientation.y = float(row["raw_hand_qy"])
            message.wrist_pose.orientation.z = float(row["raw_hand_qz"])
            message.wrist_pose.orientation.w = float(row["raw_hand_qw"])
            message.confidence = [float(row.get(field, 1.0) or 1.0) for field in CONFIDENCE_FIELDS]
            message.valid = as_bool(row.get("raw_hand_valid", "true"))
            message.gesture = int(row.get("gesture", 0) or 0)
            message.gesture_confidence = float(row.get("gesture_confidence", 0.0) or 0.0)
            message.invalid_reason = row.get("invalid_reason", "")
            self.publisher.publish(message)

    def run(self):
        while not rospy.is_shutdown():
            self.run_once()
            if not self.loop:
                break


def main():
    rospy.init_node("teleop_pose_replay")
    PoseReplay().run()


if __name__ == "__main__":
    main()
