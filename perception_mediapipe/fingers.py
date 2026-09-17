"""MediaPipe finger flexion, independent of the moving wrist-to-MCP ray.

Long digits use PIP and DIP bends; the thumb uses MCP and IP bends. Clearly
straight image chains that contradict a large predicted 3-D bend invalidate
only the finger observation. The existing controller then keeps its actuator
reference, including a held grasp, while wrist pose tracking can continue.
"""
import numpy as np

FEATURE_DEFINITION='mediapipe_digit_internal_bend_over_pi_v1'


def _digit_shape(points):
    bends=[];extension=[];lengths=[]
    for base in [1,5,9,13,17]:
        chain=points[base:base+4]
        segments=np.diff(chain,axis=0);norms=np.linalg.norm(segments,axis=1)
        if not np.all(np.isfinite(norms)) or np.min(norms)<1e-8:
            raise ValueError('DEGENERATE_FINGER_CHAIN')
        cosine=np.sum(segments[:-1]*segments[1:],axis=1)/(norms[:-1]*norms[1:])
        bends.append(float(np.clip(np.arccos(np.clip(cosine,-1,1)).sum()/np.pi,0,1)))
        extension.append(float(np.linalg.norm(chain[-1]-chain[0])/norms.sum()))
        lengths.append(float(norms.sum()))
    return np.array(bends),np.array(extension),np.array(lengths)


def observe_fingers(world_landmarks,image_landmarks,confidence):
    result=dict(contract_version=1,feature_definition=FEATURE_DEFINITION,valid=False,
                flexion=[0.]*5,confidence=0.,invalid_reason='FINGER_GEOMETRY_UNAVAILABLE')
    try:
        world=np.asarray(world_landmarks,dtype=float);image=np.asarray(image_landmarks,dtype=float)
        if world.shape!=(21,3) or image.shape!=(21,2) or not np.all(np.isfinite(world)) or not np.all(np.isfinite(image)):
            raise ValueError('INVALID_FINGER_LANDMARKS')
        bend,_,_=_digit_shape(world)
        projected,extension,lengths=_digit_shape(image)
        clear_straight=(extension[1:]>.97)&(projected[1:]<.15)
        disagreement=float(np.mean(bend[1:]-projected[1:]))
        result['diagnostics']=dict(internal_bend_3d=bend.tolist(),internal_bend_2d=projected.tolist(),
            image_extension=extension.tolist(),image_chain_length_px=lengths.tolist(),
            straight_long_finger_count=int(clear_straight.sum()),mean_excess_3d_bend=disagreement)
        # These are observation checks, not translation/rotation mode switches.
        # Thresholds were checked on the recorded open motions and natural fists.
        if sum(lengths[1:]>=12)<3:
            raise ValueError('FINGERS_TOO_SMALL_OR_FORESHORTENED')
        if sum(clear_straight)>=3 and disagreement>.20:
            raise ValueError('3D_BEND_CONTRADICTS_STRAIGHT_IMAGE_HOLD_GRIP')
        result.update(valid=True,flexion=bend.tolist(),confidence=float(np.clip(confidence,0,1)),invalid_reason='')
    except (ValueError,FloatingPointError) as exc:
        result['invalid_reason']=str(exc)
    return result
