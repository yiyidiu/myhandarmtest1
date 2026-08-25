#!/usr/bin/env python3
"""Print the configured camera-motion to robot-base velocity mapping."""

from pathlib import Path
import sys

import numpy as np
import yaml


PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from handarm_moveit_demo.shared_teleop_core import (  # noqa: E402
    RelativePoseMapper, so3_exp, so3_log,
)


def main():
    config = yaml.safe_load((PACKAGE / "config/shared_teleop.yaml").read_text(
        encoding="utf-8"))
    mapping = config["mapping"]
    control = config["control"]
    mapper = RelativePoseMapper(
        mapping["translation_matrix"], mapping["rotation_matrix"],
        mapping["translation_gain"], mapping["rotation_gain"],
        control["maximum_relative_translation_m"],
        np.deg2rad(control["maximum_relative_rotation_deg"]),
    )
    cases = [
        ("hand image-right / camera +X", [0.01, 0, 0, 0, 0, 0]),
        ("hand image-left  / camera -X", [-0.01, 0, 0, 0, 0, 0]),
        ("hand image-down  / camera +Y", [0, 0.01, 0, 0, 0, 0]),
        ("hand image-up    / camera -Y", [0, -0.01, 0, 0, 0, 0]),
        ("hand away camera / camera +Z", [0, 0, 0.01, 0, 0, 0]),
        ("hand near camera / camera -Z", [0, 0, -0.01, 0, 0, 0]),
        ("rotate MANO/tool-local +X", [0, 0, 0, 0.10, 0, 0]),
        ("rotate MANO/tool-local +Y", [0, 0, 0, 0, 0.10, 0]),
        ("rotate MANO/tool-local +Z", [0, 0, 0, 0, 0, 0.10]),
    ]
    print("configured status:", mapping.get("status", "UNKNOWN"))
    print("control mode:", control["mode"])
    print("output order: base relative target [dx dy dz droll dpitch dyaw]")
    for label, raw in cases:
        target_position, target_rotation = mapper.map(
            raw[:3], so3_exp(raw[3:]), np.zeros(3), np.eye(3))
        output = np.concatenate((target_position, so3_log(target_rotation)))
        print("{:<36} -> {}".format(
            label, np.array2string(output, precision=4, suppress_small=True)))


if __name__ == "__main__":
    main()
