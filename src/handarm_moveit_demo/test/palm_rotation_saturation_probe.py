#!/usr/bin/env python3
"""Gazebo regression: feasible yaw must remain executable at the roll limit."""
import json,math,time,sys
import numpy as np
from scipy.spatial.transform import Rotation
from pathlib import Path
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'src/handarm_moveit_demo/scripts'))
from finger_retargeting_gazebo_validator import FingerRetargetingGazeboValidator
from handarm_moveit_demo.hamer_input_contract import identity_reference_token
rospy.init_node('palm_rotation_saturation_probe',anonymous=True)
if not rospy.get_param('/use_sim_time', False) or not rospy.get_param('/palm_rotation_saturation_probe_enabled', False):
 raise RuntimeError('Use only the dedicated palm_rotation_saturation.launch Gazebo fixture')
source=FingerRetargetingGazeboValidator('127.0.0.1',5010);source.wait_ready(60)
rows=[];phase='before_C';last_trend={}
def cb(key):
 def f(m):
  global last_trend
  if key=='joints':d=dict(q=dict(zip(m.name,m.position)),v=dict(zip(m.name,m.velocity)))
  else:
   try:d=json.loads(m.data)
   except ValueError:d=m.data
  if key=='trend':last_trend=d
  rows.append(dict(phase=phase,wall=time.monotonic(),sim=rospy.Time.now().to_sec(),topic=key,data=d))
 return f
rospy.Subscriber('/joint_states',JointState,cb('joints'),queue_size=500)
for key,topic in [('trend','/shared_teleop/trend_diagnostics'),('geometry','/shared_teleop/geometric_filter_diagnostics'),('egm','/egm_position_reference/diagnostics'),('guard','/full_robot_self_collision_guard/command_diagnostic'),('hand_guard','/full_robot_self_collision_guard/hand_diagnostic'),('motor','/handarm_sim_demo/physical_hand_diagnostics'),('finger','/shared_teleop/finger_diagnostics')]:rospy.Subscriber(topic,String,cb(key),queue_size=500)
released_roll=-48.0
for phase,duration in [('before_C',1),('calibrate',2),('roll_to_limit',8),('yaw_with_roll_backlog',6),('yaw_without_roll_backlog',6)]:
 if phase=='yaw_without_roll_backlog':
  z=Rotation.from_quat(last_trend['robot_zero_quaternion_xyzw']).as_dcm()
  r=z.T@Rotation.from_quat(last_trend['current_quaternion_xyzw']).as_dcm()
  released_roll=math.degrees(math.atan2(-r[0,1],r[0,0]))/3
 start=time.monotonic()
 while time.monotonic()-start<duration:
  t=time.monotonic()-start
  side=-min(1.0,t/2)*15 if phase=='yaw_with_roll_backlog' else -15 if phase=='yaw_without_roll_backlog' else 0
  roll=-min(1.0,t/3)*60 if phase=='roll_to_limit' else -60 if phase=='yaw_with_roll_backlog' else released_roll if phase=='yaw_without_roll_backlog' else 0
  rot=Rotation.from_rotvec([math.radians(side),0,0]).as_dcm()@Rotation.from_rotvec([0,0,math.radians(roll)]).as_dcm()
  p=source.packet([.1]*5,phase!='before_C',2,wrist_position=[0,0,.55])
  p['palm_rotation_row_major']=rot.reshape(-1).tolist()
  p.update(hand_is_right=False,control_hand_is_right=False,control_reference_token=identity_reference_token(source.session,2,1,1,False)if phase!='before_C'else'')
  source.socket.sendto(json.dumps(p).encode(),('127.0.0.1',5010));time.sleep(1/26)
 print(phase,flush=True)
output=root/'.runtime/palm_rotation_regression'/('result_'+str(time.time_ns())+'.json')
output.parent.mkdir(parents=True,exist_ok=True)
output.write_text(json.dumps(dict(scope='Gazebo-only regression: unreachable roll must not erase feasible yaw; side target recovery disabled to isolate the velocity solver.',rows=rows)))
phase_rows=[r for r in rows if r['phase']=='yaw_with_roll_backlog' and r['topic']=='trend' and r['data'].get('current_quaternion_xyzw')]
if not phase_rows:raise RuntimeError('No measured robot orientation was recorded')
d=phase_rows[-1]['data'];zero=Rotation.from_quat(d['robot_zero_quaternion_xyzw']).as_dcm()
def angles(key):
 r=zero.T@Rotation.from_quat(d[key]).as_dcm()
 return np.degrees([math.atan2(-r[1,2],r[2,2]),math.asin(np.clip(r[0,2],-1,1)),math.atan2(-r[0,1],r[0,0])])
actual,target=angles('current_quaternion_xyzw'),angles('target_quaternion_xyzw')
q6=[r['data']['actual'][5]for r in rows if r['topic']=='egm' and isinstance(r['data'],dict)]
if not q6:raise RuntimeError('No actual joint feedback was recorded')
print('Target side/tilt/roll (deg):',target,'Actual:',actual,'Evidence:',output,flush=True)
if abs(target[0]-actual[0])>2 or abs(target[1]-actual[1])>2:
 raise RuntimeError('Feasible yaw failed to track while roll remained limited')
if min(q6)<-math.radians(150) or max(q6)>math.radians(150):
 raise RuntimeError('Joint 6 exceeded its operating limit')
