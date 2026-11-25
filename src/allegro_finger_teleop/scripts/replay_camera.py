#!/usr/bin/env python3
"""Replay the same serialized D455 packet into the normal local input port."""
import gzip
import json
import socket
import time
import rospy
from geometry_msgs.msg import PoseStamped

def main():
    rospy.init_node("replay_camera")
    port = rospy.get_param("~port")
    targets = [0]
    def count(_message):
        targets[0] += 1
    rospy.Subscriber("/shared_teleop/pose_target", PoseStamped, count, queue_size=5)
    while not rospy.is_shutdown() and not rospy.get_param("/fr3_allegro/fingers_ready", False):
        time.sleep(.1)
    if rospy.is_shutdown():
        return
    with gzip.open(rospy.get_param("~recording"),"rt",encoding="utf-8") as f, socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
        start, first_stamp = time.monotonic(), None
        for line in f:
            row = json.loads(line)
            recorded_session = "packet" in row
            pkt = row["packet"] if recorded_session else row
            stamp = row["time"] if recorded_session else pkt["stamp"]
            if first_stamp is None:
                first_stamp = stamp
            offset = stamp-first_stamp
            time.sleep(max(0., start+offset-time.monotonic()))
            if rospy.is_shutdown():
                return
            source_capture = pkt.get("capture_monotonic_s", pkt["stamp"])
            latency = pkt.get("timing", {}).get("capture_to_publish_s", 0.) if recorded_session else 0.
            pkt["capture_monotonic_s"] = time.monotonic()-max(0., latency)
            ref = pkt.get("control_reference_pose")
            if ref is not None and "capture_monotonic_s" in ref:
                ref["capture_monotonic_s"] = pkt["capture_monotonic_s"] + (ref["capture_monotonic_s"]-source_capture)
            raw = json.dumps(pkt,separators=(",",":"))
            sock.sendto(raw.encode(),("127.0.0.1",port))
    time.sleep(.15)
    rospy.set_param("/fr3_allegro/replay_arm_frames",targets[0])
    rospy.set_param("/fr3_allegro/replay_complete",True)
    rospy.loginfo("记录回放结束，腕部目标 %d 帧。Ctrl+C 退出。",targets[0])
    rospy.spin()

if __name__=="__main__":
    main()
