#!/usr/bin/env python3
"""Run and record the required simulated hand command cycles."""

import csv
import datetime
import json
import math
import os
import sys

import rospkg
import rospy
from gazebo_msgs.srv import GetLinkState
from std_msgs.msg import Bool

PACKAGE_PATH = rospkg.RosPack().get_path("handarm_sim_demo")
sys.path.insert(0, os.path.join(PACKAGE_PATH, "scripts"))

from hand_commander import HandCommander


SEQUENCES = {
    "RELEASE_GRASP_RELEASE": ["RELEASE", "GRASP", "RELEASE"],
}
TIP_LINKS = ["robot::f1link3", "robot::f2link2", "robot::f3link3"]


def pairwise_distances(points):
    values = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            values.append(
                math.sqrt(sum((points[i][k] - points[j][k]) ** 2 for k in range(3)))
            )
    return values


def main():
    rospy.init_node("run_hand_cycle_tests")
    rospy.loginfo(
        "[hand] Waiting for Gazebo, robot and trajectory controllers (MoveIt is not required)..."
    )
    if not rospy.wait_for_message(
        "/handarm_sim_demo/startup_ready", Bool, timeout=90.0
    ).data:
        raise SystemExit(7)
    config_path = rospy.get_param("~hand_config")
    cycles = int(rospy.get_param("~cycles", 3))
    commander = HandCommander(config_path)
    rospy.wait_for_service("/gazebo/get_link_state", timeout=15.0)
    rospy.loginfo(
        "[hand] Ready. Running %d sequence(s) x %d cycle(s).",
        len(SEQUENCES),
        cycles,
    )
    get_link = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
    rows = []
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    for sequence, commands in SEQUENCES.items():
        for cycle in range(1, cycles + 1):
            rospy.loginfo(
                "[hand] Sequence %s, cycle %d/%d started.",
                sequence,
                cycle,
                cycles,
            )
            for step, command in enumerate(commands, 1):
                rospy.loginfo(
                    "[hand] %s cycle %d step %d/%d: %s",
                    sequence,
                    cycle,
                    step,
                    len(commands),
                    command,
                )
                result = commander.command(command)
                tips = []
                for link in TIP_LINKS:
                    # Gazebo collapses the fixed handbase/tool0 chain in this URDF,
                    # so handbase_link is not exposed as a reference body.  Pairwise
                    # tip distances are frame invariant; query the available world
                    # frame instead of silently accepting a failed reference lookup.
                    response = get_link(link, "world")
                    if not response.success:
                        result["success"] = False
                        result["failure_reason"] = "tip link query failed"
                        break
                    p = response.link_state.pose.position
                    tips.append([p.x, p.y, p.z])
                result.update(
                    run_id=run_id,
                    sequence=sequence,
                    cycle=cycle,
                    step=step,
                    tip_positions_world_m=tips,
                    tip_pairwise_distances_m=pairwise_distances(tips)
                    if len(tips) == 3
                    else [],
                )
                if len(tips) == 3:
                    mean_tip_distance = sum(result["tip_pairwise_distances_m"]) / 3.0
                    if command == "GRASP":
                        kinematic_pass = mean_tip_distance <= 0.070
                    elif command == "RELEASE":
                        kinematic_pass = mean_tip_distance >= 0.110
                    else:
                        kinematic_pass = False
                    result["mean_tip_pairwise_distance_m"] = mean_tip_distance
                    result["kinematic_command_pass"] = kinematic_pass
                    result["success"] = bool(result["success"] and kinematic_pass)
                    if not kinematic_pass:
                        result["failure_reason"] = "finger-tip geometry did not match command"
                rows.append(result)
                if not result["success"]:
                    break
            if not rows[-1]["success"]:
                break
        if not rows[-1]["success"]:
            break
    results_dir = os.path.abspath(
        os.path.join(PACKAGE_PATH, "..", "..", "results", "sim_baseline")
    )
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "hand_cycles_{}.json".format(run_id))
    with open(json_path, "w", encoding="utf-8") as stream:
        json.dump(
            {
                "run_id": run_id,
                "cycles_per_sequence": cycles,
                "rows": rows,
                "all_success": len(rows) == len(SEQUENCES) * cycles * 3
                and all(row["success"] for row in rows),
                "command_scope": "GRASP_RELEASE_ONLY",
            },
            stream,
            indent=2,
            sort_keys=True,
        )
    csv_path = os.path.join(results_dir, "hand_cycles.csv")
    fields = [
        "run_id", "sequence", "cycle", "step", "command", "success",
        "execution_time_s", "target_joint_positions", "actual_joint_positions",
        "active_joint_errors_rad", "mimic_joint_errors_rad", "mimic_relation_pass",
        "kinematic_command_pass", "mean_tip_pairwise_distance_m",
        "tip_pairwise_distances_m", "failure_reason",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(row[field], sort_keys=True)
                    if isinstance(row.get(field), (dict, list))
                    else row.get(field)
                    for field in fields
                }
            )
    rospy.loginfo("[hand] DONE. Results: %s and %s", json_path, csv_path)
    rospy.loginfo(
        "[hand] Gazebo remains open. Press Ctrl-C in this terminal when finished."
    )
    success = len(rows) == len(SEQUENCES) * cycles * 3 and all(
        row["success"] for row in rows
    )
    raise SystemExit(0 if success else 7)


if __name__ == "__main__":
    main()
