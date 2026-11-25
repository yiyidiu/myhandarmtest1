#!/usr/bin/env python3
"""Gazebo-only latest-speed publisher; no joint-position integration or leash."""
import json
import threading
import time
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String

class DirectJointVelocityAdapter:
    def __init__(self):
        if not rospy.get_param('/use_sim_time', False):
            raise RuntimeError('This uncalibrated execution model is simulation only')
        # Simulation ablation: use the same bounded velocity ramp when an
        # approval is revoked. No rejected target continues to be followed.
        self.coordinated_invalid_stop = bool(rospy.get_param('~coordinated_invalid_stop', False))
        self.lock = threading.Lock()
        self.target = np.zeros(6)
        self.output = np.zeros(6)
        self.vmax = np.array([4.36332]*3 + [5.58505]*2 + [7.33038])
        self.amax = np.array([8.,8.,8.,12.,12.,16.])
        self.names = ['joint_%d' % i for i in range(1,7)]
        self.safe = False
        self.safe_received = self.state_received = -1e9
        self.command_stamp = -1e9
        self.wall_expiry = self.sim_expiry = -1e9
        self.last_tick = None
        self.tick_sequence = 0
        self.pub = rospy.Publisher('/abbarm_velocity_controller/command', Float64MultiArray, queue_size=1)
        self.execution_state = rospy.Publisher('/direct_velocity/execution_state', Float64MultiArray, queue_size=1)
        self.diag = rospy.Publisher('/direct_velocity/diagnostics', String, queue_size=10)
        rospy.Subscriber('/egm_position_reference/collision_checked_joint_velocity', Float64MultiArray,
                         self.command, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/full_robot_self_collision_guard/command_safe', Bool,
                         self.approval, queue_size=1, tcp_nodelay=True)
        rospy.Subscriber('/joint_states', JointState, self.state, queue_size=1, tcp_nodelay=True)
        self.timer = rospy.Timer(rospy.Duration(.004), self.tick)
        rospy.on_shutdown(self.shutdown)

    def command(self, msg):
        if len(msg.data) not in (6,7,8) or not np.all(np.isfinite(msg.data)):
            return
        with self.lock:
            self.target = np.clip(msg.data[:6], -self.vmax, self.vmax)
            self.command_stamp = rospy.get_time()
            self.wall_expiry = msg.data[6] if len(msg.data)>6 else time.time()+.1
            self.sim_expiry = msg.data[7] if len(msg.data)>7 else self.command_stamp+.1

    def approval(self, msg):
        with self.lock:
            self.safe, self.safe_received = msg.data, time.monotonic()

    def state(self, msg):
        if all(n in msg.name for n in self.names):
            with self.lock:
                self.state_received = time.monotonic()

    def tick(self, event):
        now = rospy.get_time()
        with self.lock:
            dt = 0. if self.last_tick is None else max(0., min(.02, now-self.last_tick))
            self.last_tick = now
            valid = (self.safe and time.monotonic()-self.safe_received<.25
                     and time.monotonic()-self.state_received<.25
                     and now-self.command_stamp<.1
                     and now<self.sim_expiry and time.time()<self.wall_expiry)
            if valid or self.coordinated_invalid_stop:
                effective_target = self.target if valid else np.zeros(6)
                delta = effective_target-self.output
                active = np.abs(delta)>1e-12
                fraction = min(1., float(np.min((self.amax*dt)[active]/np.abs(delta[active])))) if np.any(active) else 1.
                self.output += max(0.,fraction)*delta
            else:
                # Request zero immediately on revoked/expired approval. The physical
                # velocity servo still has finite response and continues braking.
                self.output.fill(0.)
            self.pub.publish(Float64MultiArray(data=self.output.tolist()))
            self.tick_sequence += 1
            # Same locked tick as the issued command; these are command-side
            # values, not measured joint velocity. Invalid input targets zero.
            snapshot_target = self.target if valid else np.zeros(6)
            self.execution_state.publish(Float64MultiArray(data=
                [now] + self.output.tolist() + snapshot_target.tolist() +
                [self.sim_expiry, float(valid), float(self.coordinated_invalid_stop),
                 float(self.tick_sequence)]))
            self.diag.publish(String(data=json.dumps({
                'stamp':now,'mode':'DIRECT_JOINT_VELOCITY_NO_POSITION_REFERENCE',
                'target_velocity':self.target.tolist(),'issued_velocity':self.output.tolist(),
                'valid':bool(valid),'command_age':now-self.command_stamp,
                'sim_expiry':self.sim_expiry,
                'tick_sequence':self.tick_sequence,
                'coordinated_invalid_stop':self.coordinated_invalid_stop,
                'braking_on_invalid':bool(not valid and self.coordinated_invalid_stop),
                'effective_target_velocity':(self.target if valid else np.zeros(6)).tolist(),
                'rate_hz':250.,'maximum_acceleration':self.amax.tolist()})))

    def shutdown(self):
        self.timer.shutdown()
        self.pub.publish(Float64MultiArray(data=[0.]*6))

if __name__=='__main__':
    rospy.init_node('direct_joint_velocity_adapter')
    DirectJointVelocityAdapter()
    rospy.spin()
