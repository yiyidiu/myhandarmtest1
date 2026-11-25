"""Compare the deployed four-finger mapping with the operator-selected November 23 version."""
import subprocess
import types
import unittest
import numpy as np
from test_retarget import human_hand, OPEN, PROJECT, load_robot
from test_fingertip_graph import opposition, folded_fist, retained_ring_pose
from retarget import Retargeter

class RestoredFingers(unittest.TestCase):
    def test_all_sixteen_joint_targets_on_identical_sequences(self):
        path = "src/allegro_finger_teleop/scripts/retarget.py"
        source = subprocess.check_output(["git","-C",str(PROJECT),"show","c4d4420:"+path])
        archived = types.ModuleType("retarget_20261123")
        exec(compile(source,path,"exec"),archived.__dict__)
        neutral = human_hand()
        model = load_robot()
        new, old = (cls(model,OPEN,neutral) for cls in (Retargeter,archived.Retargeter))
        q, q_old = np.array(OPEN), np.array(OPEN)
        goals = [opposition(g) for g in ((0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2))]
        goals += [folded_fist(), retained_ring_pose(1.,0.), retained_ring_pose(0.,1.), neutral]
        for goal in goals:
            for i in range(24):
                joints = q.copy()
                q = new.step(goal,q,.04,measured=joints)
                q_old = old.step(goal,q_old,.04,measured=joints)
                np.testing.assert_array_equal(q,q_old)
                self.assertEqual(new.gesture, old.gesture)
