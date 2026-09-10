#!/usr/bin/env python3
"""Passive host/ROS clock bridge and controller-received camera pose trace."""
import csv
import json
from pathlib import Path
import signal
import sys
import threading
import time

import rospy
from handarm_moveit_demo.msg import HamerHandPose
from rosgraph_msgs.msg import Clock


def main():
    directory = Path(sys.argv[1]).resolve()
    stop = threading.Event()
    def stop_recording(*_args):
        stop.set()
        rospy.signal_shutdown("feedback recorder stopped")
    signal.signal(signal.SIGINT, stop_recording)
    signal.signal(signal.SIGTERM, stop_recording)
    clock_file = (directory / "host_ros_clock.csv").open("x", newline="")
    hand_file = (directory / "camera_hand_pose.csv").open("x", newline="")
    events = (directory / "control_reference_events.jsonl").open("x", buffering=1)
    clocks, hands = csv.writer(clock_file), csv.writer(hand_file)
    clocks.writerow(["host_monotonic_ns", "host_wall_ns", "ros_now_ns", "latest_sim_clock_ns",
        "sim_clock_receive_monotonic_ns", "sim_clock_receive_wall_ns"])
    hands.writerow(["host_receive_monotonic_ns", "host_receive_wall_ns", "ros_receipt_ns", "source_camera_time_s",
        "source_capture_sequence", "capture_to_publish_s", "valid", "control_enabled", "reference_token",
        "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw", "invalid_reason", "inference_executed", "model_inference_s"])
    lock = threading.Lock()
    state = dict(clock_rows=0, sim_clock_messages=0, camera_rows=0, last_reference_token="",
        pose_role="Camera pose received by the controller, including the pinned pipeline's existing filtering/fusion; not unprocessed HaMeR network output.",
        source_time_role="Original camera clock from HamerHandPose.source_timestamp; kept separate from ROS receipt and host callback time.",
        simulation_time_role="Direct /clock observations with host receipt times, independent of whether /use_sim_time was set when this recorder started. Empty until a /clock message is received.")
    latest_sim_clock = ["", "", ""]

    def sim_clock_callback(message):
        mono, wall = time.monotonic_ns(), time.time_ns()
        with lock:
            latest_sim_clock[:] = [message.clock.to_nsec(), mono, wall]
            state["sim_clock_messages"] += 1

    def hand_callback(message):
        mono, wall = time.monotonic_ns(), time.time_ns()
        p, q = message.wrist_pose.position, message.wrist_pose.orientation
        with lock:
            if stop.is_set() or hand_file.closed:
                return
            hands.writerow([mono, wall, message.header.stamp.to_nsec(), message.source_timestamp,
                message.source_capture_sequence, message.capture_to_publish_s, int(message.valid), int(message.control_enabled),
                message.control_reference_token, p.x, p.y, p.z, q.x, q.y, q.z, q.w, message.invalid_reason,
                int(message.inference_executed), message.model_inference_s])
            state["camera_rows"] += 1
            if message.control_reference_token != state["last_reference_token"]:
                state["last_reference_token"] = message.control_reference_token
                events.write(json.dumps(dict(event="ROS_RECEIVED_REFERENCE_TOKEN_CHANGE", host_monotonic_ns=mono,
                    host_wall_ns=wall, ros_receipt_ns=message.header.stamp.to_nsec(),
                    token=message.control_reference_token, enabled=message.control_enabled)) + "\n")

    subscriber = None
    sim_subscriber = None
    try:
        rospy.init_node("teleop_feedback_clock", anonymous=True, disable_signals=True)
        subscriber = rospy.Subscriber("/shared_teleop/hamer_pose", HamerHandPose, hand_callback, queue_size=100)
        sim_subscriber = rospy.Subscriber("/clock", Clock, sim_clock_callback, queue_size=10)
        (directory / "clock_ready.json").write_text(json.dumps(dict(host_monotonic_ns=time.monotonic_ns(), host_wall_ns=time.time_ns())) + "\n")
        while not stop.is_set() and not rospy.is_shutdown():
            with lock:
                clocks.writerow([time.monotonic_ns(), time.time_ns(), rospy.Time.now().to_nsec()] + latest_sim_clock)
                state["clock_rows"] += 1
                if state["clock_rows"] % 50 == 0:
                    clock_file.flush()
                    hand_file.flush()
            stop.wait(.02)
    finally:
        if subscriber is not None:
            subscriber.unregister()
        if sim_subscriber is not None:
            sim_subscriber.unregister()
        with lock:
            clock_file.close()
            hand_file.close()
            events.close()
        (directory / "clock_and_hand_summary.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
