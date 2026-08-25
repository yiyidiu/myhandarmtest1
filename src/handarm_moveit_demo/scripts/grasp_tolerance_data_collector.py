#!/usr/bin/env python3
"""Next-stage real grasp-tolerance CSV collector framework (records only)."""

import csv
import json
from pathlib import Path
import threading
import time

import rospy
from std_msgs.msg import String


FIELDS = [
    "recorded_at", "trial_id", "operator_id", "object_id", "object_category",
    "object_size_x_m", "object_size_y_m", "object_size_z_m", "object_mass_kg",
    "object_material", "grasp_type", "side_direction", "commanded_assist_strength",
    "actual_assist_strength", "planned_grasp_center_pose_base",
    "actual_grasp_center_pose_base", "position_error_x_m", "position_error_y_m",
    "position_error_z_m", "orientation_error_rotvec_x_rad",
    "orientation_error_rotvec_y_rad", "orientation_error_rotvec_z_rad",
    "arm_joint_positions_rad", "arm_joint_velocities_radps", "hand_configuration",
    "hand_joint_positions_rad", "hand_joint_efforts", "fingertip_contact_states",
    "strain_gauge_raw", "strain_gauge_calibrated", "close_duration_s",
    "contact_stable_duration_s", "lift_command_m", "measured_lift_m",
    "object_slip_translation_m", "object_slip_rotation_rad", "operator_cancelled_assist",
    "reverse_override_duration_s", "servo_collision_status", "joint_limit_status",
    "workspace_limit_status", "emergency_stop", "grasp_success", "failure_reason",
    "notes", "source_bag_or_log", "hardware_calibration_id",
]

REQUIRED = {
    "trial_id", "object_id", "grasp_type", "actual_grasp_center_pose_base",
    "hand_joint_positions_rad", "grasp_success", "hardware_calibration_id",
}


class GraspToleranceCollector:
    def __init__(self):
        output = Path(rospy.get_param("~output_csv")).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        self.handle = output.open("x", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=FIELDS)
        self.writer.writeheader(); self.handle.flush()
        self.lock = threading.Lock(); self.rows = 0
        self.topic = rospy.get_param("~observation_topic", "/shared_teleop/grasp_trial_observation")
        rospy.Subscriber(self.topic, String, self.callback, queue_size=10)
        rospy.on_shutdown(self.close)
        rospy.logwarn("Grasp tolerance collector only records supplied measurements; it sends no robot/hand commands")
        rospy.loginfo("Waiting for measured trial JSON on %s; output=%s", self.topic, output)

    def callback(self, message):
        try:
            payload = json.loads(message.data)
            if payload.get("measurement_status") != "MEASURED_REAL_OR_EXPLICIT_SIM":
                raise ValueError("measurement_status must explicitly identify measured input")
            missing = sorted(field for field in REQUIRED if field not in payload)
            if missing:
                raise ValueError("missing required fields: " + ",".join(missing))
            unknown = sorted(set(payload)-set(FIELDS)-{"measurement_status"})
            if unknown:
                raise ValueError("unknown fields: " + ",".join(unknown))
            row = {field: payload.get(field, "") for field in FIELDS}
            row["recorded_at"] = row["recorded_at"] or time.time()
            for field, value in list(row.items()):
                if isinstance(value, (list, dict)):
                    row[field] = json.dumps(value, separators=(",", ":"))
            with self.lock:
                self.writer.writerow(row); self.handle.flush(); self.rows += 1
            rospy.loginfo("Recorded measured grasp-tolerance trial %s", payload["trial_id"])
        except Exception as exc:
            rospy.logwarn("Grasp trial observation rejected without writing: %s", exc)

    def close(self):
        with self.lock:
            if not self.handle.closed:
                self.handle.flush(); self.handle.close()


def main():
    rospy.init_node("grasp_tolerance_data_collector")
    GraspToleranceCollector()
    rospy.spin()


if __name__ == "__main__":
    main()
