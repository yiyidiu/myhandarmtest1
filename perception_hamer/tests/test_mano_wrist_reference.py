#!/usr/bin/env python3

from pathlib import Path
import math
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from perception_hamer.src.mano_wrist_reference import (  # noqa: E402
    build_mano_wrist_definition,
    estimate_mano_wrist_frame,
    find_boundary_edges,
    ordered_largest_boundary_loop,
)


def _rotation_xyz(x_degrees, y_degrees, z_degrees):
    x, y, z = np.radians([x_degrees, y_degrees, z_degrees])
    rx = np.array([
        [1, 0, 0], [0, math.cos(x), -math.sin(x)],
        [0, math.sin(x), math.cos(x)]])
    ry = np.array([
        [math.cos(y), 0, math.sin(y)], [0, 1, 0],
        [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([
        [math.cos(z), -math.sin(z), 0],
        [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def _angle_degrees(rotation):
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _synthetic_mano_geometry():
    vertices = np.zeros((778, 3), dtype=np.float64)
    angles = 2.0 * np.pi * np.arange(16) / 16.0
    # Elliptical open wrist boundary and an inner capped ring.  The only mesh
    # boundary is therefore the 16 outer vertices.
    vertices[:16] = np.column_stack(
        (0.030 * np.cos(angles), 0.020 * np.sin(angles), np.zeros(16))
    )
    vertices[16:32] = np.column_stack(
        (0.018 * np.cos(angles), 0.012 * np.sin(angles), 0.010*np.ones(16))
    )
    vertices[32] = [0.0, 0.0, 0.020]
    faces = []
    for index in range(16):
        following = (index + 1) % 16
        faces.append([index, following, 16 + following])
        faces.append([index, 16 + following, 16 + index])
        faces.append([16 + index, 16 + following, 32])
    joints = np.zeros((21, 3), dtype=np.float64)
    joints[0] = [0.0, 0.0, -0.015]
    joints[5] = [0.0, 0.025, 0.080]
    joints[9] = [0.0, 0.0, 0.090]
    joints[17] = [0.0, -0.025, 0.080]
    return vertices, joints, np.asarray(faces, dtype=np.int64)


class ManoWristReferenceTest(unittest.TestCase):
    def test_topology_discovers_exact_16_vertex_wrist_opening(self):
        vertices, joints, faces = _synthetic_mano_geometry()
        loop = ordered_largest_boundary_loop(find_boundary_edges(faces))
        self.assertEqual(len(loop), 16)
        definition = build_mano_wrist_definition(vertices, joints, faces, True)
        self.assertEqual(len(definition.wrist_loop), 16)

    def test_rigid_six_dof_motion_recovers_ring_centre_and_rotation(self):
        vertices, joints, faces = _synthetic_mano_geometry()
        definition = build_mano_wrist_definition(vertices, joints, faces, True)
        rotation = _rotation_xyz(18.0, -23.0, 31.0)
        translation = np.array([0.12, -0.08, 0.34])
        current = (1.15 * (rotation @ vertices.T)).T + translation
        estimate = estimate_mano_wrist_frame(current, definition)
        self.assertTrue(estimate.valid, estimate.failure_reason)
        np.testing.assert_allclose(
            estimate.origin,
            1.15 * rotation @ definition.center + translation,
            atol=1.0e-9,
        )
        np.testing.assert_allclose(
            estimate.rotation @ definition.frame.T, rotation, atol=1.0e-7
        )

    def test_one_deformed_ring_vertex_does_not_dominate_orientation(self):
        vertices, joints, faces = _synthetic_mano_geometry()
        definition = build_mano_wrist_definition(vertices, joints, faces, True)
        rotation = _rotation_xyz(-12.0, 20.0, 27.0)
        current = (rotation @ vertices.T).T + [0.1, 0.2, 0.7]
        current[definition.wrist_loop[3]] += [0.030, -0.025, 0.020]
        estimate = estimate_mano_wrist_frame(current, definition)
        self.assertTrue(estimate.valid, estimate.failure_reason)
        recovered_motion = estimate.rotation @ definition.frame.T
        self.assertLess(_angle_degrees(rotation.T @ recovered_motion), 3.0)
        weights = np.asarray(estimate.quality["vertex_weights"])
        self.assertLess(weights[3], np.median(weights))

    def test_live_finger_joint_changes_cannot_turn_wrist_ring_axes(self):
        vertices, joints, faces = _synthetic_mano_geometry()
        definition = build_mano_wrist_definition(vertices, joints, faces, True)
        first = estimate_mano_wrist_frame(vertices, definition)
        # No live joints are accepted by estimate_mano_wrist_frame; changing
        # finger articulation therefore cannot rotate its control frame.
        changed_joints = joints.copy()
        changed_joints[[5, 9, 17]] += [[0.2, 0.1, -0.1]] * 3
        second = estimate_mano_wrist_frame(vertices.copy(), definition)
        self.assertTrue(first.valid and second.valid)
        np.testing.assert_allclose(first.rotation, second.rotation, atol=1.0e-12)
        self.assertFalse(second.as_dict()["finger_joints_used_for_live_axes"])

    def test_left_and_right_source_reflections_recover_same_physical_motion(self):
        right_vertices, right_joints, faces = _synthetic_mano_geometry()
        left_vertices = right_vertices.copy(); left_vertices[:, 0] *= -1.0
        left_joints = right_joints.copy(); left_joints[:, 0] *= -1.0
        right = build_mano_wrist_definition(
            right_vertices, right_joints, faces, True)
        left = build_mano_wrist_definition(
            left_vertices, left_joints, faces, False)
        motion = _rotation_xyz(11.0, 17.0, -19.0)
        right_result = estimate_mano_wrist_frame(
            (motion @ right_vertices.T).T, right)
        left_result = estimate_mano_wrist_frame(
            (motion @ left_vertices.T).T, left)
        right_motion = right_result.rotation @ right.frame.T
        left_motion = left_result.rotation @ left.frame.T
        np.testing.assert_allclose(right_motion, motion, atol=1.0e-7)
        np.testing.assert_allclose(left_motion, motion, atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
