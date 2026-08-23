#!/usr/bin/env python3

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.palm_frame import (  # noqa: E402
    EVIDENCE_SCOPE,
    JOINT_PALM_INDICES,
    MANO_JOINT_PALM_FRAME,
    MANO_RIGID_VERTEX_PALM_FRAME,
    MIRROR_X,
    PalmFrameError,
    PalmFrameSession,
    RAW_GLOBAL_ORIENT,
    RigidPalmVertexConfig,
    align_quaternion_sign,
    compare_palm_frame_methods,
    load_rigid_palm_vertex_config,
    mano_joint_palm_frame,
    mano_rigid_vertex_palm_frame,
    mirror_canonical_rotation_to_source,
    project_to_so3,
    raw_global_orient_baseline,
    require_so3,
    rotation_distance_rad,
    rotation_matrix_to_quaternion_xyzw,
)


def rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def base_joints():
    joints = np.zeros((21, 3), dtype=np.float64)
    joints[0] = [0.0, 0.0, 0.0]
    joints[5] = [1.0, 2.0, 0.0]
    joints[9] = [0.3, 2.2, 0.0]
    joints[13] = [-0.3, 2.1, 0.0]
    joints[17] = [-1.0, 2.0, 0.0]
    # Deliberately large, asymmetric articulated finger points.  The P4 frame
    # must ignore them.
    for index in set(range(21)) - set(JOINT_PALM_INDICES):
        joints[index] = [index * 0.2, -index * 0.1, index * 0.3]
    return joints


def small_vertex_config():
    return RigidPalmVertexConfig(
        mano_vertex_count=18,
        wrist=(0, 1, 2),
        distal_palm=(3, 4, 5),
        radial_palm=(6, 7, 8),
        ulnar_palm=(9, 10, 11),
        rigid_palm=(0, 1, 2, 3, 4, 5, 12, 13, 14),
        evidence_scope=EVIDENCE_SCOPE,
        source_path="synthetic_test_config",
        mano_right_sha256="test",
        minimum_root_skinning_weight=0.98,
    )


def base_vertices():
    vertices = np.zeros((18, 3), dtype=np.float64)
    vertices[0:3] = [[-0.2, 0.0, -0.1], [0.0, 0.0, 0.1], [0.2, 0.0, 0.0]]
    vertices[3:6] = [[-0.2, 2.0, -0.1], [0.0, 2.0, 0.1], [0.2, 2.0, 0.0]]
    vertices[6:9] = [[1.0, 0.8, -0.1], [1.0, 1.0, 0.1], [1.0, 1.2, 0.0]]
    vertices[9:12] = [[-1.0, 0.8, -0.1], [-1.0, 1.0, 0.1], [-1.0, 1.2, 0.0]]
    vertices[12:15] = [[-0.5, 1.0, -0.1], [0.0, 1.0, 0.2], [0.5, 1.0, -0.1]]
    vertices[15:18] = [[100, 100, 100], [-100, 50, 20], [40, -90, 30]]
    return vertices


class SO3ContractTest(unittest.TestCase):
    def test_near_rotation_is_orthogonalized(self):
        noisy = rotation_z(0.4)
        noisy[0, 0] += 1e-3
        projected = project_to_so3(noisy)
        require_so3(projected)
        self.assertAlmostEqual(np.linalg.det(projected), 1.0, places=9)

    def test_reflection_and_nan_are_rejected(self):
        with self.assertRaises(PalmFrameError):
            project_to_so3(np.diag([-1.0, 1.0, 1.0]))
        invalid = np.eye(3)
        invalid[0, 0] = np.nan
        with self.assertRaises(PalmFrameError):
            project_to_so3(invalid)

    def test_quaternion_sign_continuity(self):
        quaternion = rotation_matrix_to_quaternion_xyzw(rotation_z(2.9))
        aligned = align_quaternion_sign(-quaternion, quaternion)
        np.testing.assert_allclose(aligned, quaternion)
        self.assertGreater(float(np.dot(aligned, quaternion)), 0.0)

    def test_two_sided_left_transform_stays_in_so3(self):
        rotation = rotation_z(0.7)
        left = mirror_canonical_rotation_to_source(rotation, False)
        np.testing.assert_allclose(left, MIRROR_X @ rotation @ MIRROR_X)
        require_so3(left)


class JointPalmFrameTest(unittest.TestCase):
    def test_joint_frame_is_so3_and_ignores_finger_articulation(self):
        joints = base_joints()
        first = mano_joint_palm_frame(joints, True)
        modified = joints.copy()
        for index in set(range(21)) - set(JOINT_PALM_INDICES):
            modified[index] += [1000.0, -2000.0, 3000.0]
        second = mano_joint_palm_frame(modified, True)
        np.testing.assert_allclose(first.rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(second.rotation, first.rotation)
        np.testing.assert_allclose(second.origin, first.origin)
        self.assertFalse(first.quality["uses_fingertips"])
        self.assertTrue(first.control_allowed)

    def test_left_right_mirror_canonical_convention(self):
        rotation = rotation_z(0.35)
        translation = np.array([2.0, 3.0, 4.0])
        right_points = base_joints() @ rotation.T + translation
        left_points = (MIRROR_X @ right_points.T).T
        right = mano_joint_palm_frame(right_points, True)
        left = mano_joint_palm_frame(left_points, False)
        np.testing.assert_allclose(right.rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(
            left.rotation, MIRROR_X @ right.rotation @ MIRROR_X, atol=1e-12
        )
        np.testing.assert_allclose(left.origin, MIRROR_X @ right.origin)

    def test_degenerate_joint_geometry_rejected(self):
        with self.assertRaises(PalmFrameError):
            mano_joint_palm_frame(np.zeros((21, 3)), True)


class RigidVertexPalmFrameTest(unittest.TestCase):
    def test_default_config_is_frozen_and_development_scoped(self):
        config = load_rigid_palm_vertex_config()
        self.assertEqual(config.mano_vertex_count, 778)
        self.assertEqual(config.evidence_scope, EVIDENCE_SCOPE)
        self.assertGreaterEqual(len(config.rigid_palm), 60)
        self.assertTrue(set(config.wrist).issubset(set(config.rigid_palm)))

    def test_vertex_frame_ignores_nonselected_vertices(self):
        config = small_vertex_config()
        vertices = base_vertices()
        first = mano_rigid_vertex_palm_frame(vertices, True, config)
        modified = vertices.copy()
        modified[15:18] *= 10000.0
        second = mano_rigid_vertex_palm_frame(modified, True, config)
        np.testing.assert_allclose(first.rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(second.rotation, first.rotation)
        np.testing.assert_allclose(second.origin, first.origin)
        self.assertFalse(first.quality["uses_articulated_finger_vertices"])
        self.assertEqual(first.quality["evidence_scope"], EVIDENCE_SCOPE)

    def test_vertex_frame_transforms_rigidly(self):
        config = small_vertex_config()
        rotation = rotation_z(-0.5)
        translation = np.array([0.2, -0.1, 3.0])
        transformed = base_vertices() @ rotation.T + translation
        result = mano_rigid_vertex_palm_frame(transformed, True, config)
        np.testing.assert_allclose(result.rotation, rotation, atol=1e-12)


class MethodAndSessionTest(unittest.TestCase):
    def test_raw_global_orient_is_baseline_only(self):
        result = raw_global_orient_baseline(rotation_z(0.2), True)
        self.assertEqual(result.method, RAW_GLOBAL_ORIENT)
        self.assertFalse(result.control_allowed)
        self.assertTrue(result.valid)

    def test_all_three_methods_compare_and_stay_so3(self):
        results = compare_palm_frame_methods(
            rotation_z(0.1),
            base_joints(),
            base_vertices(),
            True,
            small_vertex_config(),
        )
        self.assertEqual(
            set(results),
            {RAW_GLOBAL_ORIENT, MANO_JOINT_PALM_FRAME, MANO_RIGID_VERTEX_PALM_FRAME},
        )
        for estimate in results.values():
            require_so3(estimate.rotation)
        self.assertAlmostEqual(
            rotation_distance_rad(
                results[MANO_JOINT_PALM_FRAME].rotation,
                results[MANO_RIGID_VERTEX_PALM_FRAME].rotation,
            ),
            0.0,
            places=10,
        )

    def test_betas_are_frozen_and_change_is_rejected(self):
        session = PalmFrameSession(MANO_JOINT_PALM_FRAME)
        common = dict(
            global_orient=np.eye(3),
            joints=base_joints(),
            vertices=base_vertices(),
            is_right=True,
        )
        first = session.update(betas=np.zeros(10), **common)
        self.assertTrue(first.valid)
        self.assertTrue(first.reacquired)
        changed = session.update(betas=np.ones(10) * 1e-3, **common)
        self.assertFalse(changed.valid)
        self.assertEqual(changed.reason, "betas_changed_within_session")
        recovered = session.update(betas=np.zeros(10), **common)
        self.assertTrue(recovered.valid)
        self.assertTrue(recovered.reacquired)
        np.testing.assert_array_equal(session.frozen_betas, np.zeros(10))

    def test_invalid_geometry_has_no_stale_pose_and_reacquires(self):
        session = PalmFrameSession(MANO_JOINT_PALM_FRAME)
        common = dict(
            global_orient=np.eye(3),
            vertices=base_vertices(),
            betas=np.zeros(10),
            is_right=True,
        )
        valid = session.update(joints=base_joints(), **common)
        self.assertTrue(valid.valid)
        invalid_joints = base_joints()
        invalid_joints[5, 0] = np.nan
        invalid = session.update(joints=invalid_joints, **common)
        self.assertFalse(invalid.valid)
        self.assertIsNone(invalid.rotation)
        self.assertIsNone(invalid.quaternion_xyzw)
        recovered = session.update(joints=base_joints(), **common)
        self.assertTrue(recovered.valid)
        self.assertTrue(recovered.reacquired)

    def test_handedness_switch_requires_new_session(self):
        session = PalmFrameSession(MANO_JOINT_PALM_FRAME)
        first = session.update(
            global_orient=np.eye(3),
            joints=base_joints(),
            vertices=base_vertices(),
            betas=np.zeros(10),
            is_right=True,
        )
        self.assertTrue(first.valid)
        switched = session.update(
            global_orient=np.eye(3),
            joints=(MIRROR_X @ base_joints().T).T,
            vertices=(MIRROR_X @ base_vertices().T).T,
            betas=np.zeros(10),
            is_right=False,
        )
        self.assertFalse(switched.valid)
        self.assertEqual(switched.reason, "handedness_changed_requires_new_session")

    def test_documented_hamer_result_contract_is_consumed(self):
        result = SimpleNamespace(
            global_orient=np.eye(3),
            pred_keypoints_3d_source_camera_axes=base_joints(),
            pred_vertices_source_camera_axes=base_vertices(),
            betas=np.zeros(10),
            is_right=True,
        )
        session = PalmFrameSession(MANO_JOINT_PALM_FRAME)
        estimate = session.update_from_hamer(result)
        self.assertTrue(estimate.valid)
        self.assertTrue(estimate.reacquired)
        self.assertTrue(estimate.betas_frozen)


if __name__ == "__main__":
    unittest.main()
