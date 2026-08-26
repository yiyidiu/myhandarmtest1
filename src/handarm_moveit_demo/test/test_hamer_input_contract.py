#!/usr/bin/env python3

import unittest

import numpy as np

from handarm_moveit_demo.hamer_input_contract import (
    HamerPacketContract,
    InputWatchdog,
    ReferenceTokenInterlock,
    authoritative_camera_c_gate_lock,
    identity_reference_token,
)


def packet(sequence=1, valid=True):
    value = {
        "schema": "handarm_hamer_pose_v1",
        "session_id": "live-test",
        "sequence": sequence,
        "stamp": 123.5 + sequence,
        "frame_id": "camera_color_optical_frame",
        "valid": bool(valid),
        "invalid_reason": "" if valid else "no_real_hand",
        "hand_identity_present": True,
        "hand_is_right": True,
        "presence_generation": 4,
        "active_hand_generation": 2,
        "control_enabled": True,
        "control_reference_epoch": 3,
        "control_identity_present": True,
        "control_presence_generation": 4,
        "control_active_hand_generation": 2,
        "control_hand_is_right": True,
        "control_reference_token": identity_reference_token(
            "live-test", 3, 4, 2, True
        ),
        "gesture": 0,
        "gesture_confidence": 0.0,
    }
    if valid:
        value.update({
            "wrist_position_m": [0.1, -0.2, 0.7],
            "palm_rotation_row_major": np.eye(3).reshape(-1).tolist(),
            "confidence": [0.9] * 6,
        })
    return value


def add_timing(value, capture_to_publish_s=0.08):
    value["timing"] = {
        "contract_version": 1,
        "capture_sequence": 120,
        "dropped_capture_frames": 1,
        "capture_to_loop_start_s": 0.01,
        "capture_to_publish_s": capture_to_publish_s,
        "inference_executed": True,
        "inference_call_s": 0.052,
        "model_inference_s": 0.046,
        "postprocess_s": 0.018,
    }
    return value


class HamerPacketContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = HamerPacketContract("camera_color_optical_frame")

    def test_enabled_pose_requires_identity_bound_token(self):
        normalized = self.contract.validate(packet())
        self.assertTrue(normalized["observation_valid"])
        self.assertTrue(normalized["control_enabled"])
        self.assertTrue(normalized["hand_is_right"])
        np.testing.assert_allclose(normalized["position"], [0.1, -0.2, 0.7])

    def test_legacy_epoch_only_token_is_rejected(self):
        value = packet()
        value["control_reference_token"] = "live-test:3"
        with self.assertRaisesRegex(ValueError, "invalid_control_reference_token"):
            self.contract.validate(value)

    def test_observed_identity_must_match_control_identity(self):
        value = packet()
        value["hand_is_right"] = False
        with self.assertRaisesRegex(
            ValueError, "control_identity_does_not_match_observation"
        ):
            self.contract.validate(value)

    def test_invalid_heartbeat_requires_no_pose_geometry(self):
        normalized = self.contract.validate(packet(valid=False))
        self.assertFalse(normalized["observation_valid"])
        self.assertEqual(normalized["invalid_reason"], "no_real_hand")
        np.testing.assert_allclose(normalized["position"], np.zeros(3))
        np.testing.assert_allclose(
            normalized["quaternion"], [0.0, 0.0, 0.0, 1.0]
        )

    def test_rejected_packet_does_not_consume_sequence(self):
        bad = packet(sequence=8)
        bad["control_reference_token"] = "bad"
        with self.assertRaises(ValueError):
            self.contract.validate(bad)
        accepted = self.contract.validate(packet(sequence=8))
        self.assertEqual(accepted["sequence"], 8)

    def test_nonfinite_gesture_confidence_is_rejected(self):
        value = packet()
        value["gesture_confidence"] = float("nan")
        with self.assertRaisesRegex(ValueError, "gesture_confidence must be finite"):
            self.contract.validate(value)

    def test_gesture_must_fit_ros_uint8(self):
        value = packet()
        value["gesture"] = 256
        with self.assertRaisesRegex(ValueError, "gesture must fit uint8"):
            self.contract.validate(value)

    def test_producer_timing_is_validated_and_normalized(self):
        normalized = self.contract.validate(add_timing(packet()))
        self.assertTrue(normalized["timing_contract_present"])
        self.assertEqual(normalized["source_capture_sequence"], 120)
        self.assertEqual(normalized["dropped_capture_frames"], 1)
        self.assertAlmostEqual(normalized["capture_to_publish_s"], 0.08)
        self.assertAlmostEqual(normalized["inference_call_s"], 0.052)

    def test_pipeline_latency_gate_rejects_stale_measurement_without_consuming_sequence(self):
        contract = HamerPacketContract(
            "camera_color_optical_frame", maximum_pipeline_latency_s=0.20
        )
        with self.assertRaisesRegex(ValueError, "source_pipeline_latency_exceeded"):
            contract.validate(add_timing(packet(sequence=9), 0.21))
        accepted = contract.validate(add_timing(packet(sequence=9), 0.19))
        self.assertEqual(accepted["sequence"], 9)

    def test_valid_timed_observation_requires_real_inference_measurement(self):
        value = add_timing(packet())
        value["timing"]["inference_executed"] = False
        value["timing"]["inference_call_s"] = 0.0
        value["timing"]["model_inference_s"] = 0.0
        with self.assertRaisesRegex(
            ValueError, "valid observation requires measured inference timing"
        ):
            self.contract.validate(value)

    def test_strict_live_contract_rejects_valid_packet_without_timing(self):
        contract = HamerPacketContract(
            "camera_color_optical_frame",
            maximum_pipeline_latency_s=0.20,
            require_timing_contract=True,
        )
        with self.assertRaisesRegex(
            ValueError, "live packet requires timing contract"
        ):
            contract.validate(packet(sequence=6))
        with self.assertRaisesRegex(
            ValueError, "live packet requires timing contract"
        ):
            contract.validate(packet(sequence=6, valid=False))
        status = add_timing(packet(sequence=5, valid=False))
        status["timing"].update({
            "inference_executed": False,
            "inference_call_s": 0.0,
            "model_inference_s": 0.0,
            "postprocess_s": 0.0,
        })
        normalized_status = contract.validate(status)
        self.assertFalse(normalized_status["observation_valid"])
        accepted = contract.validate(add_timing(packet(sequence=6)))
        self.assertEqual(accepted["sequence"], 6)


class InputWatchdogTest(unittest.TestCase):
    def test_timeout_repeats_until_a_packet_is_accepted(self):
        watchdog = InputWatchdog(0.4, 0.1, start_monotonic=10.0)
        self.assertFalse(watchdog.timeout_due(10.39))
        self.assertTrue(watchdog.timeout_due(10.40))
        self.assertFalse(watchdog.timeout_due(10.45))
        self.assertTrue(watchdog.timeout_due(10.50))
        watchdog.mark_accepted(10.51)
        self.assertFalse(watchdog.timeout_due(10.90))
        self.assertTrue(watchdog.timeout_due(10.91))


class CameraCGateEvidenceTest(unittest.TestCase):
    def test_only_camera_origin_disabled_packet_satisfies_startup_edge(self):
        self.assertTrue(authoritative_camera_c_gate_lock(False, True))
        self.assertFalse(authoritative_camera_c_gate_lock(False, False))
        self.assertFalse(authoritative_camera_c_gate_lock(True, True))


class ReferenceTokenInterlockTest(unittest.TestCase):
    def test_timeout_blocks_same_token_until_new_c_epoch(self):
        interlock = ReferenceTokenInterlock()
        first = {
            "control_enabled": True,
            "control_reference_token": "live:1:p2:h3:R",
        }
        interlock.accept(first)
        self.assertEqual(
            interlock.require_new_reference(), "live:1:p2:h3:R"
        )
        with self.assertRaisesRegex(
            ValueError, "blocked_reference_token_requires_new_c"
        ):
            interlock.accept(first)
        interlock.accept({
            "control_enabled": True,
            "control_reference_token": "live:2:p2:h3:R",
        })
        self.assertIsNone(interlock.blocked_token)

    def test_disabled_status_cannot_silently_clear_blocked_token(self):
        interlock = ReferenceTokenInterlock()
        enabled = {
            "control_enabled": True,
            "control_reference_token": "live:4:p1:h1:L",
        }
        interlock.accept(enabled)
        interlock.require_new_reference()
        interlock.accept({
            "control_enabled": False,
            "control_reference_token": "",
        })
        with self.assertRaisesRegex(
            ValueError, "blocked_reference_token_requires_new_c"
        ):
            interlock.accept(enabled)

    def test_rejected_new_token_itself_is_blocked(self):
        interlock = ReferenceTokenInterlock()
        old = {
            "control_enabled": True,
            "control_reference_token": "live:1:p1:h1:R",
        }
        interlock.accept(old)
        interlock.require_new_reference("live:2:p1:h1:R")
        with self.assertRaisesRegex(
            ValueError, "blocked_reference_token_requires_new_c"
        ):
            interlock.accept({
                "control_enabled": True,
                "control_reference_token": "live:2:p1:h1:R",
            })
        with self.assertRaisesRegex(
            ValueError, "blocked_reference_token_requires_new_c"
        ):
            interlock.accept(old)
        interlock.accept({
            "control_enabled": True,
            "control_reference_token": "live:3:p1:h1:R",
        })


if __name__ == "__main__":
    unittest.main()
