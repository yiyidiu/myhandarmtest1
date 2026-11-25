"""Validate a single camera/C stream and dispatch its two observable motion channels."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from retarget import Retargeter, ThumbFilter, MIDDLE_WEIGHTS
from hand_shape import HandShape
from wrist_arm import WristArm, read_wrist
from wrist_motion import PalmMotionSupport


BASELINE = "teleop-wrist-qb0-20261120"
FRAME = "wrist_qb0_openpose21_m"


def read_landmarks(observation):
    if not isinstance(observation, dict) or observation.get("frame") != FRAME:
        raise ValueError("缺少腕环坐标系下的 21 点")
    points = np.asarray(observation.get("points"), dtype=float)
    conf = float(observation.get("confidence", 0))
    if points.shape != (21, 3) or not np.isfinite(points).all() or not 0.35 <= conf <= 1.0:
        raise ValueError("手部关键点或置信度无效")
    # Validate each phalanx. Absolute distances are MANO metres, never depth-camera translation.
    bones = np.linalg.norm(np.diff(points[1:].reshape(5, 4, 3), axis=1), axis=2)
    if np.any((bones < .006) | (bones > .11)) or np.max(np.linalg.norm(points, axis=1)) > .32:
        raise ValueError("手指骨段尺度异常")
    return points


@dataclass
class Commands:
    dt: float
    hand: Optional[np.ndarray] = None
    points: Optional[np.ndarray] = None
    arm: Optional[np.ndarray] = None
    palm_target: Optional[tuple] = None


class FingerStream:
    def __init__(self, model, open_q, arm_model=None, middle_weights=MIDDLE_WEIGHTS, require_wrist_support=False, housing_guard=None, object_pinch=None):
        self.model = model
        self.object_pinch = object_pinch
        self.housing_guard = housing_guard
        self.middle_weights = middle_weights
        self.require_wrist_support = require_wrist_support
        self.arm_model = arm_model
        self.arm_map = None
        self.last_hand_goal = None
        self.goal_velocity = np.zeros((4, 4))
        self.palm_support = None
        self.open_q = np.asarray(open_q)
        self.token = None
        self.retargeter = None
        self.session = None
        self.epoch = 0
        self.closed_epoch = 0
        self.last_capture = -np.inf
        self.last_sequence = -1
        self.last_source = None
        self.last_sim = None
        self.last_presence = None
        self.q = None
        self.reason = "等待张开左手按 C"

    def accept(self, packet, now, sim_time, measured):
        """Original hand-only interface."""
        result = self.accept_frame(packet, now, sim_time, measured)
        if result is not None and result.hand is not None:
            return result.hand, result.points, result.dt
        return None

    def filter_finger_goals(self, goal, previous, prior_goal, dt, restart):
        """Give each digit more bandwidth during sustained angular motion."""
        if restart:
            self.goal_velocity.fill(0.)
        rate = ((goal-prior_goal)/dt).reshape(4, 4)
        self.goal_velocity += (rate-self.goal_velocity)*dt/(dt+1./(2*np.pi*2.))
        cutoff = np.minimum(18., 1.5+3.*np.max(np.abs(self.goal_velocity), axis=1))
        gain = np.repeat(dt/(dt+1./(2*np.pi*cutoff)), 4)
        return previous + gain*(goal-previous)

    def accept_frame(self, packet, now, sim_time, measured, arm_measured=None, arm_reference=None):
        try:
            return self._accept(packet, now, sim_time, measured, arm_measured, arm_reference)
        except (TypeError, ValueError, KeyError, FloatingPointError, np.linalg.LinAlgError) as exc:
            self.reason = str(exc)
            return None

    def _accept(self, packet, now, sim_time, measured, arm_measured, arm_reference):
        if not isinstance(packet, dict) or packet.get("schema") != "handarm_hamer_pose_v1":
            raise ValueError("未知相机数据格式")
        timing = packet.get("timing") or {}
        if not isinstance(timing, dict):
            raise ValueError("缺少采集时间")
        capture = float(packet.get("capture_monotonic_s", -np.inf))
        if (not np.isfinite(capture) or not -0.002 <= now-capture <= .25
                or capture <= self.last_capture):
            raise ValueError("跳过过期或乱序的相机帧")
        session = packet["session_id"]
        seq = int(timing["capture_sequence"])
        epoch = int(packet["control_reference_epoch"])
        if session == self.session and seq <= self.last_sequence:
            raise ValueError("跳过重复的采集帧")
        if session != self.session:
            self.token, self.retargeter, self.closed_epoch = None, None, 0
            self.epoch = 0
        self.session, self.last_sequence, self.last_capture = session, seq, capture
        if packet.get("control_enabled") is not True:
            self.closed_epoch = max(self.closed_epoch, epoch)
            self.token = None
            if packet.get("control_gate_reason") == "HAND_IDENTITY_CHANGED_REQUIRES_NEW_C":
                raise ValueError("控制手已更换，张开左手按 C 重新标定")
            raise ValueError("等待张开左手按 C")
        fixed_hand = packet.get("control_reference_policy") == "fixed_hand"
        if fixed_hand and packet.get("hand_identity_present") is not True:
            raise ValueError("左手暂时不可见，保持当前位置，重新识别左手后继续跟随")
        if (not session or epoch <= self.closed_epoch or epoch < self.epoch or packet.get("control_identity_present") is not True
                or packet.get("hand_identity_present") is not True or packet.get("hand_is_right") is not False
                or packet.get("control_hand_is_right") is not False):
            raise ValueError("等待已标定的同一只左手")
        presence, active = int(packet["presence_generation"]), int(packet["active_hand_generation"])
        bound_presence = int(packet["control_presence_generation"])
        expected = "{}:{}:p{}:h{}:L".format(session, epoch, bound_presence, active)
        if (min(presence, bound_presence, active) < 0 or expected != packet.get("control_reference_token")
                or (epoch == self.epoch and self.token != expected)
                or presence < bound_presence or (presence != bound_presence and not fixed_hand)
                or (self.token == expected and self.last_presence is not None and presence < self.last_presence)
                or active != packet.get("control_active_hand_generation")):
            raise ValueError("手部身份或标定序号不匹配")
        if (packet.get("valid") is not True or (self.arm_model is not None and packet.get("wrist_pose_baseline") != BASELINE)
                or timing.get("inference_executed") is not True
                or not 0 <= float(timing.get("capture_to_publish_s", -1)) <= .25):
            raise ValueError("当前帧缺少有效的腕姿或手指观测")
        if not 0 <= float(packet["stamp"]) < 2**32 or not 0 <= seq < 2**32:
            raise ValueError("采集时间或帧号无效")
        if self.token != expected:
            ref = packet["control_reference_pose"]
            if not 0 <= int(ref["sequence"]) <= int(packet["sequence"]):
                raise ValueError("标定参考帧序号无效")
            if not ref.get("hand_landmarks"):
                raise ValueError("C 帧未看清张开的手，腕部可跟随；需要四指控制时张开左手再按 C")
            reference = read_landmarks(ref["hand_landmarks"])
            self.retargeter = Retargeter(self.model, self.open_q, reference, self.middle_weights)
            self.last_hand_goal = None
            self.shape = HandShape(reference, ref["hand_landmarks"], ref.get("measured_wrist_rotation_row_major"))
            self.thumb_filter = ThumbFilter(reference)
            self.arm_map = None
            self.palm_support = None
            self.q = np.clip(np.asarray(measured), self.model.lower, self.model.upper)
            self.token = expected
            self.epoch = epoch
            self.last_source, self.last_sim = None, None
            self.last_presence = None
            self.closed_epoch = epoch-1
        dt = 1/30 if self.last_sim is None else min(capture-self.last_source, sim_time-self.last_sim, .12)
        if dt <= 1e-5:
            raise ValueError("等待仿真时钟推进")
        resuming = (self.last_source is not None and
                    (capture-self.last_source > .25 or presence != self.last_presence))
        if resuming:
            # Restart the trajectory from feedback, with one ordinary frame of
            # motion budget. The human and robot poses recorded at C stay fixed.
            dt = min(dt, 1/30)
            self.shape.interrupt()
        commands = Commands(dt)
        reasons = []
        fresh_rotation = packet.get("orientation_held") is False and packet.get("orientation_channel_valid") is True
        try:
            observation = packet.get("hand_landmarks")
            points = read_landmarks(observation)
            if not fresh_rotation and observation.get("wrist_frame_measured") is not True:
                raise ValueError("等待同帧腕姿和手指观测")
            shape = self.shape.update(points, observation, packet.get("measured_wrist_rotation_row_major"))
            filtered = self.thumb_filter.update(shape, dt)
            previous = measured if resuming else self.q
            prior_goal = previous if resuming or self.last_hand_goal is None else self.last_hand_goal
            goal = self.retargeter.step(filtered, prior_goal, dt, measured=measured,
                                        observed_tip=self.shape.observed_tip, separated_tip=self.shape.separated_tip,
                                        retained_only=self.shape.edge_on)
            if self.object_pinch is not None:
                if resuming or self.last_hand_goal is None:
                    self.object_pinch.partners = ()
                goal = self.object_pinch.approach(goal, previous, self.retargeter.contacts, dt)
                goal = self.object_pinch.apply(goal, measured, shape, self.retargeter.contacts, sim_time)
            if self.shape.edge_on and self.shape.hidden_partners.any():
                held = np.repeat(self.shape.held_digits, 4)
                goal[held] = previous[held]
                self.goal_velocity.reshape(-1)[held] = 0.
                reasons.append("侧视下捏合手指身份不清，保持待确认指位")
            averaged = goal
            if self.shape.image_required:
                averaged = self.filter_finger_goals(goal, previous, prior_goal, dt,
                                                    resuming or self.last_hand_goal is None)
            limit = self.model.speed*dt
            self.q = np.clip(averaged, previous-limit, previous+limit)
            if self.housing_guard is not None:
                safe = self.housing_guard.limit_hand(previous, self.q)
                limited = np.abs(safe - self.q) > 1e-8
                goal[limited] = safe[limited]
                self.goal_velocity.reshape(-1)[limited] = 0.
                self.q = safe
            self.last_hand_goal = goal
            commands.hand, commands.points = self.q, points
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            self.shape.interrupt()
            reasons.append(str(exc))
        if self.arm_model is not None and arm_measured is not None:
            try:
                confidence = np.asarray(packet.get("confidence"), dtype=float)
                if (confidence.shape != (6,) or not np.isfinite(confidence).all() or np.any(confidence[:3] <= 0)
                        or packet.get("frame_id") != "camera_color_optical_frame"
                        or packet.get("position_source")
                        != "D455_WRIST_RING_CENTER_FROM_SAME_PIXEL_SURFACE_OFFSET"):
                    raise ValueError("缺少有效的腕环中心深度")
                held_rotation = packet.get("orientation_held") is True and np.all(confidence[3:] == 0)
                if not (held_rotation or (fresh_rotation and np.all(confidence[3:] > 0))):
                    raise ValueError("腕部旋转观测无效")
                position, rotation = read_wrist(packet)
                if self.arm_map is None:
                    self.arm_map = WristArm(self.arm_model, packet["control_reference_pose"],
                                            self.retargeter.basis, arm_measured, arm_reference)
                ref = packet["control_reference_pose"]
                uses_depth = self.require_wrist_support or "wrist_motion_support" in ref or "wrist_motion_support" in packet
                if self.palm_support is None and uses_depth:
                    self.palm_support = PalmMotionSupport(ref.get("wrist_motion_support"),
                                                          self.arm_map.hpos, self.arm_map.hrot, ref.get("wrist_palm_rays"))
                if held_rotation:
                    rotation = self.arm_map.last_rotation
                if self.palm_support is not None:
                    if resuming:
                        self.palm_support.interrupt()
                    position, rotation = self.palm_support.update(packet.get("wrist_motion_support"),
                                                                  position, rotation, capture, not held_rotation,
                                                                  packet.get("wrist_palm_rays"))
                    if self.palm_support.mode == "translation":
                        reasons.append("掌部平移可验证，保持腕部转角")
                    if not self.palm_support.available:
                        reasons.append("掌部深度点不足或不一致，保持腕部目标")
                if resuming:
                    self.arm_map.command = np.clip(arm_measured, self.arm_model.lower, self.arm_model.upper)
                commands.arm, commands.palm_target = self.arm_map.step((position, rotation), arm_measured, dt)
            except (KeyError, ValueError, TypeError) as exc:
                reasons.append(str(exc))
        self.reason = "，".join(reasons) if reasons else "正在跟随左手"
        if commands.arm is None and commands.hand is None:
            return None
        self.last_source, self.last_sim = capture, sim_time
        self.last_presence = presence
        return commands
