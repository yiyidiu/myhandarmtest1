std_msgs/Header header
# Original HaMeR/D455 clock in seconds. Header.stamp is ROS-domain receipt time.
float64 source_timestamp
geometry_msgs/Pose wrist_pose
float32[6] confidence
bool valid
uint8 gesture
float32 gesture_confidence
string invalid_reason
# Live UDP operator gate.  The camera process sets gate_present=true on every
# packet; control_enabled becomes true only after a valid C-key reference.
# The token changes on every successful C press, allowing the ROS controller
# to atomically re-capture both hand zero and the current robot tool pose.
bool control_gate_present
bool control_enabled
uint32 control_reference_epoch
string control_reference_token
