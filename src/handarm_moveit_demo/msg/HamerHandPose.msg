std_msgs/Header header
# Original HaMeR/D455 clock in seconds. Header.stamp is ROS-domain receipt time.
float64 source_timestamp
# Producer-side timing from the exact RGB-D frame used for this measurement.
# These values distinguish true camera measurements from controller replay.
bool timing_contract_present
uint64 source_capture_sequence
uint32 dropped_capture_frames
float32 capture_to_publish_s
bool inference_executed
float32 inference_call_s
float32 model_inference_s
float32 postprocess_s
# Robot-independent, continuous MANO/OpenPose finger-chain observation.
# The five entries are [thumb,index,middle,ring,pinky] accumulated flexion,
# normalized by pi. Robot morphology mapping is intentionally downstream.
bool finger_tracking_present
bool finger_tracking_valid
float32[5] finger_flexion
float32 finger_tracking_confidence
string finger_invalid_reason
geometry_msgs/Pose wrist_pose
float32[6] confidence
bool valid
uint8 gesture
float32 gesture_confidence
string invalid_reason
# Observed camera-side hand identity. Generations are monotonic for one camera
# process and prevent a new/reacquired/opposite hand from inheriting an old C.
bool hand_identity_present
bool hand_is_right
uint32 presence_generation
uint32 active_hand_generation
# Live UDP operator gate.  The camera process sets gate_present=true on every
# packet; control_enabled becomes true only after a valid C-key reference.
# The token changes on every successful C press, allowing the ROS controller
# to atomically re-capture both hand zero and the current robot tool pose.
bool control_gate_present
bool control_enabled
uint32 control_reference_epoch
string control_reference_token
