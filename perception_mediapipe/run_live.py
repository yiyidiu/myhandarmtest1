#!/usr/bin/env python3
"""MediaPipe RGB-D teleoperation with the existing C gate and UDP interface.

The legacy packet/topic names are an ABI, not a HaMeR dependency. Only generic
capture, SO(3) filtering, finger bend geometry and the operator gate are reused.
"""
import argparse, json, os, socket, sys, threading, time, uuid
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from perception_mediapipe.pose import MediaPipePose
from perception_hamer.src.d455_capture import D455Capture
from perception_hamer.src.causal_wrist_so3_filter import CausalWristSO3Filter
from perception_hamer.src.teleop_control_gate import TeleopControlGate
from perception_hamer.src.finger_observation import invalid_finger_observation


class PosePublisher:
    def __init__(self, mode):
        self.mode=mode;self.session='mediapipe_'+uuid.uuid4().hex[:12]
        self.filter=CausalWristSO3Filter()
        self.gate=TeleopControlGate(retain_reference_until_operator_action=True)
        self.last_side=None
        self._reference_reset_after=None
        self._calibration_wait_reason='waiting_for_hand_pose'

    def request_reference(self, after_sequence):
        """Recalibration starts a new measurement history while output is locked.

        The jump guard's old estimate is not a valid reference for a new C.
        Reset only on the first fresh frame AFTER the request. In particular,
        never reset the filter merely because a live motion was rejected.
        """
        self.gate.request_confirmation(after_sequence)
        self._reference_reset_after=int(after_sequence)
        self._calibration_wait_reason='waiting_for_hand_pose'

    def packet(self, pose, stamp, seq, inference_s, age_s, dropped=0):
        if self._reference_reset_after is not None and seq>self._reference_reset_after:
            self.filter.reset()
            self.last_side=None
            self._reference_reset_after=None
        p=dict(schema='handarm_hamer_pose_v1',session_id=self.session,sequence=seq,
            stamp=stamp,frame_id='camera_color_optical_frame',valid=False,
            invalid_reason=pose.get('reason','NO_HAND'),confidence=[0.]*6,
            gesture=0,gesture_confidence=0.,hand_identity_present=False,
            hand_is_right=False,presence_generation=1,active_hand_generation=1,
            finger_observation=invalid_finger_observation('NO_VALID_POSE'),
            orientation_source='MEDIAPIPE_'+self.mode.upper(),
            position_source='MEDIAPIPE_WRIST_0_ALIGNED_RGBD',
            control_reference='MEDIAPIPE_WRIST_0_PALM_BASES',
            palm_assistance_enabled=False,
            timing=dict(contract_version=1,capture_sequence=seq,dropped_capture_frames=dropped,
                capture_to_publish_s=age_s,inference_executed=True,
                inference_call_s=inference_s,model_inference_s=inference_s,
                postprocess_s=pose.get('geometry_ms',0)/1000))
        if pose['valid']:
            side=pose['is_right']
            if self.last_side != side:self.filter.reset();self.last_side=side
            rotation=np.asarray(pose['rotation']).reshape(3,3)
            filtered=self.filter.update(stamp,rotation,pose['confidence'])
            p.update(hand_identity_present=True,hand_is_right=side,
                     raw_rotation_row_major=pose['rotation'],orientation_filter=filtered.as_dict())
            if filtered.valid and age_s <= .25:
                p.update(valid=True,invalid_reason='',confidence=[pose['confidence']]*3+[filtered.confidence]*3,
                    wrist_position_m=pose['position'],palm_rotation_row_major=filtered.rotation.reshape(-1).tolist(),
                    orientation_channel_valid=True,orientation_held=False,finger_observation=pose['fingers'])
            else:p['invalid_reason']=filtered.reason if age_s<=.25 else 'PIPELINE_LATENCY_EXCEEDED'
        # Preserve the actual observation reason before the operator gate
        # changes an otherwise valid observation into a locked control packet.
        p['measurement_valid_before_C_gate']=bool(p['valid'])
        p['measurement_reason']=p['invalid_reason']
        if self.gate.confirmation_pending:
            fingers=p['finger_observation']
            if not p['valid']:
                self._calibration_wait_reason='waiting_for_hand_pose'
            elif not fingers['valid'] or np.any(np.array(fingers['flexion'][1:])>.40):
                self._calibration_wait_reason='waiting_for_open_hand'
            else:
                self._calibration_wait_reason='waiting_for_stable_pose'
        self.gate.confirm_packet(p,lambda:True)
        output=self.gate.decorate(p)
        if self.gate.enabled:status='calibrated'
        elif self.gate.confirmation_pending:status=self._calibration_wait_reason
        elif self.gate.lock_reason=='C_REFERENCE_NOT_STABLE_RETRY':status='failed_'+self._calibration_wait_reason
        else:status='waiting_for_c'
        output['calibration_status']=status
        return output


def draw(rgb,d,pose,packet,rate):
    bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    if d is not None:
        xy=d['xy'].astype(int)
        for a,b in ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                    (13,17),(17,18),(18,19),(19,20),(0,17)):
            cv2.line(bgr,tuple(xy[a]),tuple(xy[b]),(90,220,60),1,cv2.LINE_AA)
        for p in xy:cv2.circle(bgr,tuple(p),2,(70,120,255),-1)
        if pose['valid']:
            origin=xy[0];rot=np.array(pose['rotation']).reshape(3,3)
            for k,color in enumerate([(0,0,255),(0,210,0),(255,80,0)]):
                tip=origin+(rot[:2,k]*55).astype(int)
                cv2.arrowedLine(bgr,tuple(origin),tuple(tip),color,2,cv2.LINE_AA)
                cv2.putText(bgr,'XYZ'[k],tuple(tip),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
    labels={'calibrated':'CONTROL ON - zero set',
        'waiting_for_c':'PRESS C: open hand, hold still',
        'waiting_for_hand_pose':'C RECEIVED: waiting for valid hand pose',
        'waiting_for_open_hand':'C RECEIVED: waiting for open-hand reading',
        'waiting_for_stable_pose':'C RECEIVED: collecting stable pose frames'}
    state=packet.get('calibration_status','waiting_for_c')
    status=labels.get(state,'C not completed - see terminal; press C to retry')
    cv2.putText(bgr,status,(12,23),cv2.FONT_HERSHEY_SIMPLEX,.53,(40,240,40),1,cv2.LINE_AA)
    cv2.putText(bgr,f"LEFT FIXED  {packet['orientation_source']}  {rate:.1f} Hz  C: zero  Q: quit",(12,47),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
    if not packet['valid']:
        cv2.putText(bgr,packet['invalid_reason'][:64],(12,71),cv2.FONT_HERSHEY_SIMPLEX,.42,(40,120,255),1,cv2.LINE_AA)
    if pose['valid'] and not pose['fingers']['valid']:
        cv2.putText(bgr,'Finger shape uncertain: keeping previous grip',(12,94),cv2.FONT_HERSHEY_SIMPLEX,.42,(40,190,255),1,cv2.LINE_AA)
    return bgr


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['rgbd','world'],default='world')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--duration',type=float,default=3600)
    p.add_argument('--no-gui',action='store_true')
    p.add_argument('--replay',type=Path,help='comparison cache; uses original time intervals')
    p.add_argument('--replay-c-after',type=float,help='explicit simulated C for integration verification only')
    p.add_argument('--udp-port',type=int,default=5010)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    # Set affinity BEFORE Hands creates its inference worker threads.
    if hasattr(os,'sched_setaffinity') and os.cpu_count()>=12:os.sched_setaffinity(0,{8,9,10,11})
    model=MediaPipePose();publisher=PosePublisher(args.mode);udp=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    stop=threading.Event();condition=threading.Condition();slot=[None,0,None]
    capture=None;thread=None;replay_rows=None
    if args.replay:
        replay_rows=[json.loads(x) for x in open(args.replay/'mediapipe.jsonl')]
        replay_rows=[x for x in replay_rows if x['source_found']]
        meta=json.load(open(args.replay/'metadata.json'))
    else:
        capture=D455Capture(width=640,height=480,fps=30,
            record_path=os.environ.get('TELEOP_RGBD_RECORDING_FILE')).start()
        json.dump(capture.device_metadata,open(args.output/'camera_metadata.json','w'),indent=2)
        def acquire():
            try:
                if hasattr(os,'sched_setaffinity') and os.cpu_count()>=12:os.sched_setaffinity(0,{2,3})
                while not stop.is_set():
                    f=capture.wait_for_frame()
                    with condition:slot[0]=f;slot[1]+=1;condition.notify()
            except Exception as exc:
                with condition:slot[2]=exc;condition.notify()
        thread=threading.Thread(target=acquire,daemon=True);thread.start()
    began=time.monotonic();previous=0;seq=0;last_display=0;rates=[];c_requested=False
    last_calibration_status='waiting_for_c';last_epoch=0
    print('MediaPipe',args.mode,'ready. FIXED LEFT HAND. No HaMeR or palm auxiliary. Press C before control.',flush=True)
    try:
        with (args.output/'frames.jsonl').open('w',buffering=1) as log:
            while time.monotonic()-began<args.duration:
                if args.replay:
                    if seq>=len(replay_rows):break
                    row=replay_rows[seq];due=began+row['t']-replay_rows[0]['t']
                    if due>began+args.duration:break
                    while time.monotonic()<due:time.sleep(min(.02,due-time.monotonic()))
                    data=np.load(args.replay/'frames'/row['file'])
                    f=SimpleNamespace(rgb=data['rgb'],aligned_depth_raw=data['depth'],
                        depth_scale_m_per_unit=meta['depth_scale'],color_intrinsics=meta['intrinsics'],
                        color_timestamp_ms=time.time()*1000,host_monotonic_ns_frameset_received=time.monotonic_ns())
                    version=seq+1
                else:
                    with condition:
                        condition.wait_for(lambda:slot[1]>previous or slot[2] is not None,timeout=.5)
                        if slot[2] is not None:raise slot[2]
                        if slot[1]==previous:continue
                        f,version=slot[:2]
                dropped=max(0,version-previous-1);previous=version;seq+=1
                binding=publisher.gate.bound_identity
                d,elapsed=model.detect(f.rgb,None if binding is None else binding[2])
                pose=model.poses(d,f.aligned_depth_raw,f.depth_scale_m_per_unit,f.color_intrinsics,modes=(args.mode,))[args.mode]
                if args.replay_c_after is not None and not c_requested and time.monotonic()-began>=args.replay_c_after:
                    publisher.request_reference(seq);c_requested=True
                age=(time.monotonic_ns()-f.host_monotonic_ns_frameset_received)/1e9
                packet=publisher.packet(pose,f.color_timestamp_ms/1000,seq,elapsed,age,dropped)
                if packet['control_reference_epoch']!=last_epoch and packet['control_enabled']:
                    print('[标定] 零位已建立，现在可以移动手控制机械臂。',flush=True)
                    last_epoch=packet['control_reference_epoch']
                state=packet['calibration_status']
                if state.startswith('failed_') and state!=last_calibration_status:
                    reason={'failed_waiting_for_hand_pose':'当前没有连续可用的手部位姿',
                        'failed_waiting_for_open_hand':'手指读数未达到张手要求',
                        'failed_waiting_for_stable_pose':'位置或方向尚未满足原来的稳定性要求'}.get(state,state)
                    print('[标定] C 已收到，但本次未完成：'+reason+'。',flush=True)
                last_calibration_status=state
                udp.sendto(json.dumps(packet,allow_nan=False).encode(),('127.0.0.1',args.udp_port))
                now=time.monotonic();rates.append(now);rates=[t for t in rates if now-t<2]
                log.write(json.dumps(dict(packet=packet,raw_pose=pose,landmarks=None if d is None else d['xy'].tolist(),
                    landmarks_world_m=None if d is None else d['world'].tolist(),
                    handedness_policy='fixed_left',
                    detector_reported_is_right=None if d is None else d['detector_reported_is_right'],
                    applied_hand_is_right=None if d is None else d['is_right'],
                    color_frame_number=getattr(f,'color_frame_number',None),
                    elapsed_s=now-began),allow_nan=False)+'\n')
                if seq%100==0:print(f'{seq} frames; {packet["control_gate_reason"]}; capture->pose {age*1000:.1f} ms',flush=True)
                if not args.no_gui and now-last_display>=1/15:
                    cv2.imshow('MediaPipe teleoperation',draw(f.rgb,d,pose,packet,len(rates)/2));last_display=now
                key=-1 if args.no_gui else cv2.waitKey(1)&255
                if key in (ord('q'),27):break
                if key in (ord('c'),ord('C')):
                    publisher.request_reference(seq)
                    print('[标定] 已收到 C，正在用新的手部观测建立零位。',flush=True)
    finally:
        stop.set()
        if capture:capture.stop()
        if thread:thread.join(timeout=2)
        model.close();udp.close();cv2.destroyAllWindows()

if __name__=='__main__':main()
