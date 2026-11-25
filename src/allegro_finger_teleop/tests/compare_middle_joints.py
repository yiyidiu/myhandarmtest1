"""Evaluate one set of middle-joint weights on known independent human hand motions."""
import argparse
import json

import numpy as np

from test_retarget import HUMAN, OPEN, Retargeter, human_hand, load_robot
from retarget import HUMAN_MIDDLE, ROBOT_MIDDLE, ROWS
from test_wrist_hand import long_hand, thumb_sweep
from test_fingertip_graph import opposition, folded_fist, aim_finger


def evaluate(weights, validation=False):
    model = load_robot()
    neutral = human_hand()
    start_q = np.array(OPEN)
    reference = Retargeter(model, OPEN, neutral, weights)
    site_errors, free_tip_errors, flex_errors, quiet_errors = [], [], [], []
    contact_errors, wrong, observed_curl = [], [], []
    def solve(mapper, hand, q, steps=28):
        for _ in range(steps):
            q = mapper.step(hand, q, .075)
        return q
    def record_sites(q, hand):
        delta = (hand @ reference.basis)[HUMAN_MIDDLE]-reference.middle_human_open
        goal = reference.middle_robot_open + reference.middle_scale[:,None]*(reference.middle_alignment @ delta[:,:,None])[:,:,0]
        pair_delta = delta[:3]-delta[3]
        pair_goal = (reference.middle_robot_open[:3]-reference.middle_robot_open[3]
                     + reference.middle_pair_scale[:,None]*(reference.middle_pair_alignment @ pair_delta[:,:,None])[:,:,0])
        pos = model.kinematics(q)[0]
        center = pos[ROWS,ROBOT_MIDDLE]
        site_errors.append([np.linalg.norm(center[3]-goal[3])**2, np.mean(np.sum((center[:3]-goal[:3])**2,axis=1)),
                            np.mean(np.sum((center[:3]-center[3]-pair_goal)**2,axis=1))])
        return pos[:,4]
    shapes = ([.35,.94,.67],[.88,.44,.90],[1.31,1.25,1.07]) if validation else ([.47,.26,.14],[.81,.91,.70],[1.25,1.38,1.18])
    for f in range(3):
        for angles in shapes:
            h=long_hand(f, angles)
            q=solve(Retargeter(model,OPEN,neutral,weights),h,start_q)
            tips=record_sites(q,h)
            goal=reference.robot_open+reference.scale[:,None]*((h@reference.basis)[HUMAN[:,3]]-reference.human_open)
            free_tip_errors.append(np.linalg.norm(tips[f]-goal[f])**2)
            flex_errors.extend(np.abs((q-start_q).reshape(4,4)[f,1:]-angles))
            quiet_errors.append(np.abs(np.delete((q-start_q).reshape(4,4),f,axis=0)).max())
    for turn in ((-.15,.13,.32) if validation else (-.11,.22,.42)):
        h=thumb_sweep(turn); q=solve(Retargeter(model,OPEN,neutral,weights),h,start_q)
        record_sites(q,h)
        flex_errors.append(abs(q[12]-start_q[12]-turn))
        quiet_errors.append(np.abs(q[:12]-start_q[:12]).max())
    for group in ((0,), (1,), (2,), (0,1), (1,2), (0,2), (0,1,2)):
        mapper=Retargeter(model,OPEN,neutral,weights)
        h=opposition(group)
        if validation:
            # Offset the contact target relative to the tuning cases.
            h=h.copy()
            for f in group:
                aim_finger(h,HUMAN[f],h[HUMAN[f,-1]]+[.0015, .0012*(f-1), -.0022])
        q=solve(mapper,h,start_q,36)
        tips=record_sites(q,h)
        if mapper.contacts!=tuple((f,3) for f in group):wrong.append(str(group))
        contact_errors.extend([abs(np.linalg.norm(tips[f]-tips[3])-model.pad_radii[f]-model.pad_radii[3]) for f in group])
        for f in set(range(3))-set(group):
            quiet_errors.append(np.abs((q-start_q)[4*f:4*f+4]).max())
    for folded in (True,False):
        mapper=Retargeter(model,OPEN,neutral,weights); h=folded_fist(folded)
        q=solve(mapper,h,start_q,40); record_sites(q,h)
        if mapper.gesture!='power_grasp':wrong.append('fist')
        observed_curl.append(np.degrees(q[:12].reshape(3,4)[:,1:]-start_q[:12].reshape(3,4)[:,1:]).min())
        if not folded:quiet_errors.append(np.abs(q[12:]-start_q[12:]).max())
    errors=1000*np.sqrt(np.mean(site_errors,axis=0))
    return dict(weights=list(weights), site_rms_mm=errors.tolist(), aggregate_site_rms_mm=float(np.sqrt(np.mean(errors**2))),
                free_tip_rms_mm=1000*np.sqrt(np.mean(free_tip_errors)), independent_flex_mae_deg=float(np.degrees(np.mean(flex_errors))),
                max_quiet_finger_deg=float(np.degrees(max(quiet_errors))), max_pinch_pad_error_mm=1000*max(contact_errors),
                min_fist_flex_deg=float(min(observed_curl)), wrong_gestures=wrong)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('weights',nargs=3,type=float)
    ap.add_argument('--validation',action='store_true')
    args=ap.parse_args()
    print(json.dumps(evaluate(args.weights,args.validation), indent=2))
