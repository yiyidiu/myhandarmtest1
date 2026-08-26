#!/usr/bin/env python3

from pathlib import Path
import sys
import unittest

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))

from perception_hamer.src.teleop_control_gate import TeleopControlGate  # noqa: E402
from handarm_moveit_demo.hamer_input_contract import (  # noqa: E402
    HamerPacketContract,
    ReferenceTokenInterlock,
)


FINGER_FEATURE = "mano_openpose_chain_total_bend_over_pi_v1"


def observation(
    sequence,
    valid=True,
    presence_generation=4,
    active_hand_generation=2,
    hand_is_right=True,
    reason="",
):
    packet = {
        "schema": "handarm_hamer_pose_v1",
        "session_id": "fault-injection",
        "sequence": int(sequence),
        "stamp": 1000.0 + int(sequence),
        "frame_id": "camera_color_optical_frame",
        "valid": bool(valid),
        "invalid_reason": str(reason),
        "hand_identity_present": bool(valid),
        "hand_is_right": bool(hand_is_right),
        "presence_generation": int(presence_generation),
        "active_hand_generation": int(active_hand_generation),
        "gesture": 0,
        "gesture_confidence": 0.0,
        "finger_observation": {
            "contract_version": 1,
            "feature_definition": FINGER_FEATURE,
            "valid": bool(valid),
            "flexion": [0.1] * 5 if valid else [0.0] * 5,
            "confidence": 0.9 if valid else 0.0,
            "invalid_reason": str(reason),
        },
        "timing": {
            "contract_version": 1,
            "capture_sequence": int(sequence),
            "dropped_capture_frames": 0,
            "capture_to_loop_start_s": 0.01,
            "capture_to_publish_s": 0.08,
            "inference_executed": bool(valid),
            "inference_call_s": 0.05 if valid else 0.0,
            "model_inference_s": 0.04 if valid else 0.0,
            "postprocess_s": 0.01,
        },
    }
    if valid:
        packet.update({
            "wrist_position_m": [0.1, -0.2, 0.7],
            "palm_rotation_row_major": np.eye(3).reshape(-1).tolist(),
            "confidence": [0.9] * 6,
        })
    return packet


class OperatorLatchedCContractTest(unittest.TestCase):
    def test_perception_faults_hold_without_changing_c_token(self):
        gate = TeleopControlGate(
            retain_reference_until_operator_action=True
        )
        gate.confirm(4, 2, True)
        contract = HamerPacketContract(
            "camera_color_optical_frame",
            maximum_pipeline_latency_s=0.25,
            require_timing_contract=True,
            require_finger_contract=True,
        )
        interlock = ReferenceTokenInterlock()
        packets = [
            observation(1),
            observation(
                2,
                reason="causal_so3_filter:orientation_jump",
            ),
            observation(
                3,
                valid=False,
                reason="SOURCE_PIPELINE_LATENCY_EXCEEDED",
            ),
            observation(
                4,
                valid=False,
                presence_generation=31,
                active_hand_generation=4,
                hand_is_right=False,
                reason="no_real_hand",
            ),
            observation(
                5,
                presence_generation=32,
                active_hand_generation=5,
                hand_is_right=False,
            ),
        ]
        packets[2]["timing"]["capture_to_publish_s"] = 0.35

        normalized = []
        decorated = []
        for packet in packets:
            output = gate.decorate(packet)
            accepted = contract.validate(output)
            interlock.accept(accepted)
            decorated.append(output)
            normalized.append(accepted)

        self.assertTrue(all(item["control_enabled"] for item in decorated))
        self.assertEqual(
            len({item["control_reference_token"] for item in decorated}),
            1,
        )
        self.assertEqual(
            [item["observation_valid"] for item in normalized],
            [True, True, False, False, True],
        )


if __name__ == "__main__":
    unittest.main()
