"""Paired FR3/Allegro retargeting comparisons. All distances are in metres internally."""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import types

import numpy as np
from scipy.optimize import least_squares

from test_retarget import HUMAN, OPEN, PROJECT, Retargeter, human_hand, load_robot
from test_fingertip_graph import aim_finger, folded_fist, opposition, retained_ring_pose
from test_wrist_hand import READY, arm_model, long_hand, thumb_sweep
from retarget import human_features

# Keep the November 24 paper ablation reproducible independently of the live solver.
paper_version = "fr3-wrist-contact-objectives-20261124"
paper_path = "src/allegro_finger_teleop/scripts/retarget.py"
paper_source = subprocess.check_output(["git", "-C", str(PROJECT), "show", paper_version + ":" + paper_path])
paper_solver = types.ModuleType("paper_retarget")
exec(compile(paper_source, paper_path, "exec"), paper_solver.__dict__)
DEPLOYED_OBJECTIVES = paper_solver.DEPLOYED_OBJECTIVES
HandObjectives = paper_solver.HandObjectives
from wrist_arm import rotation_vector

BASE_TAG = "fr3-wrist-knuckles-20261123"
AUTHOR_HASH = "c9f9ff03cd62db0a49f7fc2183a599edf168f93073ad14381d00123f14e7de72"
AUTHOR_FILE = PROJECT.parent / "data/retargeting-xin-2026/avp_teleop_2025-01-16_20-27-43.npz"
VARIANTS = {
    "original": None,
    "repair_only": HandObjectives(distal=1.0),
    "approach_1": HandObjectives(distal=1.0, approach=1.0),
    "approach_3": HandObjectives(distal=1.0, approach=3.0),
    "approach_6": HandObjectives(distal=1.0, approach=6.0),
    "no_wrist_tip": HandObjectives(wrist_tip=0.0, distal=1.0),
    "no_opposition": HandObjectives(opposition=False, distal=1.0),
    "unscaled_pinch": HandObjectives(rescale=False, distal=1.0),
    "no_distal": HandObjectives(distal=0.0),
    "distal_1.5": HandObjectives(distal=1.5),
    "distal_2": HandObjectives(distal=2.0),
    "distal_3": HandObjectives(distal=3.0),
    "position_distal": HandObjectives(wrist_tip=1.4, distal=3.0),
    "position_distal_approach1": HandObjectives(wrist_tip=1.4, distal=3.0, approach=1.0),
    "position_distal_approach2": HandObjectives(wrist_tip=1.4, distal=3.0, approach=2.0),
    "position_distal_approach3": HandObjectives(wrist_tip=1.4, distal=3.0, approach=3.0),
    "combined_approach1": HandObjectives(wrist_tip=2.0, distal=3.0, approach=1.0),
    "combined_approach2": HandObjectives(wrist_tip=2.0, distal=3.0, approach=2.0),
    "digit_approach1": HandObjectives(wrist_tip=2.0, distal=3.0, approach=1.0, contact_priority="digit"),
    "digit_approach2": HandObjectives(wrist_tip=2.0, distal=3.0, approach=2.0, contact_priority="digit"),
    "digit_approach3": HandObjectives(wrist_tip=2.0, distal=3.0, approach=3.0, contact_priority="digit"),
    "phase_only": HandObjectives(wrist_tip=2.0, distal=3.0, contact_priority="group"),
    "phase_approach1": HandObjectives(wrist_tip=2.0, distal=3.0, approach=1.0, contact_priority="group"),
    "phase_approach2": HandObjectives(wrist_tip=2.0, distal=3.0, approach=2.0, contact_priority="group"),
    "phase_approach3": HandObjectives(wrist_tip=2.0, distal=3.0, approach=3.0, contact_priority="group"),
    "phase_approach4": HandObjectives(wrist_tip=2.0, distal=3.0, approach=4.0, contact_priority="group"),
    "phase_approach6": HandObjectives(wrist_tip=2.0, distal=3.0, approach=6.0, contact_priority="group"),
    "position2_distal3": HandObjectives(wrist_tip=2.0, distal=3.0),
    "wrist_tip_1.4": HandObjectives(wrist_tip=1.4, distal=1.0),
}


@lru_cache(maxsize=1)
def old_solver():
    path = "src/allegro_finger_teleop/scripts/retarget.py"
    source = subprocess.check_output(["git", "-C", str(PROJECT), "show", BASE_TAG + ":" + path])
    module = types.ModuleType("original_retarget")
    exec(compile(source, BASE_TAG + "/" + path, "exec"), module.__dict__)
    return module.Retargeter


def make_solver(name, model, open_hand):
    if name == "original":
        return old_solver()(model, OPEN, open_hand)
    weights = DEPLOYED_OBJECTIVES if name == "deployed" else VARIANTS[name]
    return paper_solver.Retargeter(model, OPEN, open_hand, objectives=weights)


def desired(ref, points):
    local = points @ ref.basis
    tip, direction = human_features(local)
    target = ref.robot_open + ref.scale[:, None] * (tip - ref.human_open)
    return target, (ref.align_dirs @ direction[..., None])[..., 0]


def hand_errors(model, ref, points, q):
    target, dirs = desired(ref, points)
    frame, direction = model.kinematics(q)[:2]
    tips = frame[:, 4]
    dot = np.einsum("ij,ij->i", dirs[:, 1], direction[:, 1])
    tip = 1000 * np.linalg.norm(tips - target, axis=1)
    thumb = 1000 * np.linalg.norm((tips[:3] - tips[3]) - (target[:3] - target[3]), axis=1)
    return tip, thumb, np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))), tips


def cases(held_out):
    shapes = (
        ([0.35, 0.94, 0.67], [0.88, 0.44, 0.90], [1.31, 1.25, 1.07])
        if held_out
        else ([0.47, 0.26, 0.14], [0.81, 0.91, 0.70], [1.25, 1.38, 1.18])
    )
    for f in range(3):
        for angles in shapes:
            yield "flex", long_hand(f, angles), [f]
    for turn in (-0.15, 0.13, 0.32) if held_out else (-0.11, 0.22, 0.42):
        yield "sweep", thumb_sweep(turn), [3]
    for group in ((0,), (1,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2)):
        hand = opposition(group)
        if held_out:
            for f in group:
                aim_finger(hand, HUMAN[f], hand[HUMAN[f, -1]] + [0.0015, 0.0012 * (f - 1), -0.0022])
        yield "pinch", hand, list(group)
    for folded in (True, False):
        yield "fist", folded_fist(folded), list(range(4 if folded else 3))
    if held_out:
        for i, m in ((0.6, 0.15), (0.15, 0.92), (0.90, 0.82)):
            yield "hold", retained_ring_pose(i, m), [2]


def postures(name, model, held_out):
    neutral = human_hand()
    ref = Retargeter(model, OPEN, neutral)
    rows = []
    for kind, hand, active in cases(held_out):
        sol = make_solver(name, model, neutral)
        q = np.array(OPEN)
        for _ in range(35):
            q = sol.step(hand, q, 0.06, measured=q)
        tip, relative, angle, pads = hand_errors(model, ref, hand, q)
        observed = sorted(set(active + ([3] if kind in ("pinch", "hold") else [])))
        gaps = (
            [
                1000 * abs(np.linalg.norm(pads[f] - pads[3]) - model.pad_radii[f] - model.pad_radii[3])
                for f in active
            ]
            if kind in ("pinch", "hold")
            else []
        )
        inactive = [i for i in range(4) if i not in observed]
        excursion = np.abs(np.degrees(q - np.array(OPEN))).reshape(4, 4)
        rows.append(
            dict(
                kind=kind,
                observed=observed,
                tip_mm=tip[observed].tolist(),
                distal_deg=angle[observed].tolist(),
                thumb_relative_mm=relative.tolist(),
                pad_error_mm=gaps,
                quiet_deg=excursion[inactive].max() if inactive and kind != "hold" else 0.0,
                fist_long_min_deg=excursion[:3, 1:].min() if kind == "fist" else None,
            )
        )

    def flat(key):
        return [v for r in rows for v in r[key]]

    return dict(
        postures=len(rows),
        wrist_tip_mean_mm=np.mean(flat("tip_mm")),
        thumb_relative_mean_mm=np.mean(flat("thumb_relative_mm")),
        distal_mean_deg=np.mean(flat("distal_deg")),
        pad_error_max_mm=max(flat("pad_error_mm")),
        quiet_max_deg=max(r["quiet_deg"] for r in rows),
        fist_long_min_deg=min(r["fist_long_min_deg"] for r in rows if r["kind"] == "fist"),
    )


def recording(name, model, path, source_hz=60.0):
    if hashlib.sha256(path.read_bytes()).hexdigest() not in (
        AUTHOR_HASH,
        "5c58deaa4ed5db8f1cf08d783d48df94164c9b98baeee30ab9c6b9b2cb6d62a7",
    ):
        raise ValueError("wrong author fixture")
    with np.load(path, allow_pickle=False) as archive:
        # Authors' 25-to-21 AVP mapping. Mirror the recorded right hand for our left Allegro.
        select = [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
        hand = archive["stream_right_fingers"][:, :, :3, 3][:, select].copy()
    hand[:, :, 0] *= -1
    # Same maximally extended calibration sample for every variant, fixed before scoring.
    reach = np.linalg.norm(hand[:, [4, 8, 12, 16, 20]] - hand[:, 0:1], axis=2).sum(axis=1)
    ci = int(np.argmax(reach))
    ref = Retargeter(model, OPEN, hand[ci])
    sol = make_solver(name, model, hand[ci])
    q = np.array(OPEN)
    err = []
    labels = []
    gaps = []
    travel = []
    for h in hand[::2]:
        last = q.copy()
        q = sol.step(h, q, 2 / source_hz, measured=q)
        tip, relative, angle, pads = hand_errors(model, ref, h, q)
        err.append([tip.mean(), relative.mean(), angle.mean()])
        hg = np.linalg.norm(h[[8, 12, 16]] - h[4], axis=1) / ref.width
        if hg.min() < 0.27:
            f = int(np.argmin(hg))
            gap = 1000 * (np.linalg.norm(pads[f] - pads[3]) - model.pad_radii[f] - model.pad_radii[3])
            gaps.append(max(0.0, gap))
        labels.append(sol.gesture)
        travel.append(np.abs(q - last).sum())
    return dict(
        input_frames=len(hand),
        sampled_frames=len(err),
        calibration_frame=ci,
        source_hz=source_hz,
        means=list(np.mean(err, axis=0)),
        human_close_pair_frames=len(gaps),
        close_pair_gap_median_mm=float(np.median(gaps)) if gaps else None,
        close_pair_under_2mm_fraction=float(np.mean(np.array(gaps) < 2.0)) if gaps else None,
        total_joint_travel_rad=float(sum(travel)),
        labels={k: labels.count(k) for k in sorted(set(labels))},
    )


def global_test(model):
    # Isolate global anchoring with identical, already-solved finger joints in both arms.
    # Then solve a separate 23-joint analogue of Eq.(6) on the same static poses.
    arm = arm_model()
    start, R0, _ = arm.fk(READY)
    neutral = human_hand()
    ref = Retargeter(model, OPEN, neutral)
    out = {key: [] for key in ("fixed_wrist", "movable_wrist_same_fingers", "joint_23")}
    for kind, h, active in cases(True):
        if kind not in ("pinch", "flex", "sweep"):
            continue
        q = np.array(OPEN)
        sol = make_solver("repair_only", model, neutral)
        for _ in range(35):
            q = sol.step(h, q, 0.06, measured=q)
        target, directions = desired(ref, h)
        actual_wrist = start - R0 @ arm.wrist_to_palm

        def measure(arm_q, hand_q):
            p, r, _ = arm.fk(arm_q)
            current_wrist = p - r @ arm.wrist_to_palm
            o, ds = model.kinematics(hand_q)[:2]
            tips = o[:, 4]
            global_tips = p + tips @ r.T
            global_target = start + target @ R0.T
            return dict(
                wrist_mm=1000 * np.linalg.norm(current_wrist - actual_wrist),
                wrist_deg=np.degrees(np.linalg.norm(rotation_vector(r @ R0.T))),
                global_thumb_mm=1000 * np.linalg.norm(global_tips[3] - global_target[3]),
                global_tip_mean_mm=1000 * np.linalg.norm(global_tips - global_target, axis=1).mean(),
                relative_tip_mean_mm=1000 * np.linalg.norm(tips - target, axis=1).mean(),
                distal_deg=np.degrees(np.arccos(np.clip((ds[:, 1] * directions[:, 1]).sum(1), -1, 1))).mean(),
            )

        out["fixed_wrist"].append(measure(READY, q))
        # Translation-only optimum with exact wrist orientation; small wrist-position penalty.
        thumb = model.kinematics(q)[0][3, 4]
        goal = start + R0 @ (target[3] - thumb) / (1.0 + 0.15**2)

        def arm_cost(a):
            p, r, _ = arm.fk(a)
            return np.r_[p - goal, 0.24 * rotation_vector(R0 @ r.T), 0.0005 * (a - READY)]

        qa = least_squares(arm_cost, READY, bounds=(arm.lower, arm.upper), max_nfev=35).x
        out["movable_wrist_same_fingers"].append(measure(qa, q))
        # Four paper objectives adapted to our calibrated anthropometry, marker centres and pad radii.
        local = h @ ref.basis
        vectors = local[HUMAN[:3, -1]] - local[HUMAN[3, -1]]
        d = np.linalg.norm(vectors, axis=1)
        width = ref.width
        weight = 1.0 / (1.0 + np.exp(np.clip(14 * (d / width - 0.64), -50, 50)))
        gain = (ref.scale[:3] + ref.scale[3]) / 2
        minimum = model.pad_radii[:3] + model.pad_radii[3]
        low, high = 0.35 * width, 0.72 * width
        scaled = np.where(
            d > high,
            np.maximum(minimum, d * gain),
            minimum
            + np.clip((d - low) / (high - low), 0.0, 1.0) * (np.maximum(minimum, gain * high) - minimum),
        )
        pinch_goal = vectors / np.maximum(d[:, None], 1e-7) * scaled[:, None]

        def cost(x):
            p, r, _ = arm.fk(x[:7])
            o, ds = model.kinematics(x[7:])[:2]
            tips = o[:, 4]
            global_thumb = p + r @ tips[3] - (start + R0 @ target[3])
            return np.r_[
                2.0 * global_thumb,
                0.035 * rotation_vector(R0 @ r.T),
                (tips - target).ravel(),
                (0.06 * (ds[:, 1] - directions[:, 1])).ravel(),
                (2.5 * np.sqrt(weight[:, None]) * (tips[:3] - tips[3] - pinch_goal)).ravel(),
                0.005 * (x[7:] - OPEN),
                0.002 * (x[:7] - READY),
            ]

        result = least_squares(
            cost,
            np.r_[qa, q],
            bounds=(np.r_[arm.lower, model.lower], np.r_[arm.upper, model.upper]),
            max_nfev=38,
            ftol=1e-5,
            xtol=1e-5,
            gtol=2e-6,
        )
        out["joint_23"].append(measure(result.x[:7], result.x[7:]))
    return {
        k: dict(count=len(v), **{m: float(np.mean([r[m] for r in v])) for m in v[0]}) for k, v in out.items()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=list(VARIANTS) + ["global", "deployed"])
    parser.add_argument("--recording", type=Path, default=AUTHOR_FILE)
    parser.add_argument("--source-hz", type=float, choices=(30.0, 60.0), default=60.0)
    parser.add_argument(
        "--recording-only", action="store_true", help="score only the chosen author hand trajectory"
    )
    args = parser.parse_args()
    model = load_robot()
    if args.variant == "global":
        res = global_test(model)
    elif args.recording_only:
        res = dict(avp=recording(args.variant, model, args.recording, args.source_hz))
    else:
        res = dict(
            tune=postures(args.variant, model, False),
            validation=postures(args.variant, model, True),
            avp=recording(args.variant, model, args.recording, args.source_hz),
        )
    print(json.dumps(res, indent=2, allow_nan=False))
