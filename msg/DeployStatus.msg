# Controller states mirror LowLevelController::RobotState.
uint8 CONTROLLER_STATE_IDLE=0
uint8 CONTROLLER_STATE_FIX_STAND=1
uint8 CONTROLLER_STATE_PRE_IDLE=2
uint8 CONTROLLER_STATE_HIGH_CONTROLLER=3

uint8 controller_state
bool safety_ok
bool fix_stand_ready

# Incremented after each successful active-policy reset.
uint64 policy_reset_sequence
bool policy_output_valid

# Empty until a policy update throws; cleared by a successful policy reset.
string last_fault
