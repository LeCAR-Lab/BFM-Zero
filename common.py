"""
Common configuration values for BFM-Zero MuJoCo simulation.

This module defines PD control gains, action scales, and default joint positions
based on BFM-Zero/config/policy/motivo_newG1.yaml
"""

import numpy as np

# Policy joint names in order (29 joints)
POLICY_JOINT_NAMES = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 
    'left_elbow_joint', 
    'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 
    'right_elbow_joint', 
    'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
]

# Action scales for each joint (in policy_joint_names order)
ACTION_SCALES = np.array([
    0.222001498914,  # left_hip_pitch_joint
    0.22200157,      # left_hip_roll_joint
    0.54754699,      # left_hip_yaw_joint
    0.35066156,      # left_knee_joint
    0.43857802,      # left_ankle_pitch_joint
    0.43857802,      # left_ankle_roll_joint
    0.222001498914,  # right_hip_pitch_joint
    0.22200157,      # right_hip_roll_joint
    0.54754699,      # right_hip_yaw_joint
    0.35066156,      # right_knee_joint
    0.43857802,      # right_ankle_pitch_joint
    0.43857802,      # right_ankle_roll_joint
    0.54754699,      # waist_yaw_joint
    0.43857802,      # waist_roll_joint
    0.43857802,      # waist_pitch_joint
    0.43857802,      # left_shoulder_pitch_joint
    0.43857802,      # left_shoulder_roll_joint
    0.43857802,      # left_shoulder_yaw_joint
    0.43857802,      # left_elbow_joint
    0.43857802,      # left_wrist_roll_joint
    0.07450086,      # left_wrist_pitch_joint
    0.07466888,      # left_wrist_yaw_joint
    0.43857802,      # right_shoulder_pitch_joint
    0.43857802,      # right_shoulder_roll_joint
    0.43857802,      # right_shoulder_yaw_joint
    0.43857802,      # right_elbow_joint
    0.43857802,      # right_wrist_roll_joint
    0.07450086,      # right_wrist_pitch_joint
    0.07450086,      # right_wrist_yaw_joint
], dtype=np.float32)

# PD control proportional gains (Kp) for each joint (in policy_joint_names order)
KP_GAINS = np.array([
    99.09843,   # left_hip_pitch_joint (.*_hip_pitch_joint)
    99.0984,    # left_hip_roll_joint (.*_hip_roll_joint)
    40.1792,    # left_hip_yaw_joint (.*_hip_yaw_joint)
    99.0984,    # left_knee_joint (.*_knee_joint)
    28.5012,    # left_ankle_pitch_joint (.*ankle_pitch_joint)
    28.5012,    # left_ankle_roll_joint (.*ankle_roll_joint)
    99.09843,   # right_hip_pitch_joint (.*_hip_pitch_joint)
    99.0984,    # right_hip_roll_joint (.*_hip_roll_joint)
    40.1792,    # right_hip_yaw_joint (.*_hip_yaw_joint)
    99.0984,    # right_knee_joint (.*_knee_joint)
    28.5012,    # right_ankle_pitch_joint (.*ankle_pitch_joint)
    28.5012,    # right_ankle_roll_joint (.*ankle_roll_joint)
    40.1792,    # waist_yaw_joint (waist_yaw_joint)
    28.5012,    # waist_roll_joint (waist_roll_joint)
    28.5012,    # waist_pitch_joint (waist_pitch_joint)
    14.2506,    # left_shoulder_pitch_joint (.*_shoulder_.*)
    14.2506,    # left_shoulder_roll_joint (.*_shoulder_.*)
    14.2506,    # left_shoulder_yaw_joint (.*_shoulder_.*)
    14.2506,    # left_elbow_joint (.*_elbow_joint)
    14.2506,    # left_wrist_roll_joint (.*_wrist_roll_joint)
    16.7783,    # left_wrist_pitch_joint (.*_wrist_pitch_joint)
    16.7783,    # left_wrist_yaw_joint (.*_wrist_yaw_joint)
    14.2506,    # right_shoulder_pitch_joint (.*_shoulder_.*)
    14.2506,    # right_shoulder_roll_joint (.*_shoulder_.*)
    14.2506,    # right_shoulder_yaw_joint (.*_shoulder_.*)
    14.2506,    # right_elbow_joint (.*_elbow_joint)
    14.2506,    # right_wrist_roll_joint (.*_wrist_roll_joint)
    16.7783,    # right_wrist_pitch_joint (.*_wrist_pitch_joint)
    16.7783,    # right_wrist_yaw_joint (.*_wrist_yaw_joint)
], dtype=np.float32)

# PD control derivative gains (Kd) for each joint (in policy_joint_names order)
KD_GAINS = np.array([
    6.30880,    # left_hip_pitch_joint (.*_hip_pitch_joint)
    6.3088,     # left_hip_roll_joint (.*_hip_roll_joint)
    2.5579,     # left_hip_yaw_joint (.*_hip_yaw_joint)
    6.3088,     # left_knee_joint (.*_knee_joint)
    1.8145,     # left_ankle_pitch_joint (.*ankle_pitch_joint)
    1.8145,     # left_ankle_roll_joint (.*ankle_roll_joint)
    6.30880,    # right_hip_pitch_joint (.*_hip_pitch_joint)
    6.3088,     # right_hip_roll_joint (.*_hip_roll_joint)
    2.5579,     # right_hip_yaw_joint (.*_hip_yaw_joint)
    6.3088,     # right_knee_joint (.*_knee_joint)
    1.8145,     # right_ankle_pitch_joint (.*ankle_pitch_joint)
    1.8145,     # right_ankle_roll_joint (.*ankle_roll_joint)
    2.5579,     # waist_yaw_joint (waist_yaw_joint)
    1.8145,     # waist_roll_joint (waist_roll_joint)
    1.8145,     # waist_pitch_joint (waist_pitch_joint)
    0.9072,     # left_shoulder_pitch_joint (.*_shoulder_.*)
    0.9072,     # left_shoulder_roll_joint (.*_shoulder_.*)
    0.9072,     # left_shoulder_yaw_joint (.*_shoulder_.*)
    0.9072,     # left_elbow_joint (.*_elbow_joint)
    0.9072,    # left_wrist_roll_joint (.*_wrist_roll_joint)
    1.0681,    # left_wrist_pitch_joint (.*_wrist_pitch_joint)
    1.0681,    # left_wrist_yaw_joint (.*_wrist_yaw_joint)
    0.9072,     # right_shoulder_pitch_joint (.*_shoulder_.*)
    0.9072,     # right_shoulder_roll_joint (.*_shoulder_.*)
    0.9072,     # right_shoulder_yaw_joint (.*_shoulder_.*)
    0.9072,     # right_elbow_joint (.*_elbow_joint)
    0.9072,    # right_wrist_roll_joint (.*_wrist_roll_joint)
    1.0681,    # right_wrist_pitch_joint (.*_wrist_pitch_joint)
    1.0681,    # right_wrist_yaw_joint (.*_wrist_yaw_joint)
], dtype=np.float32)

# Default joint positions for each joint (in policy_joint_names order)
# Only hip_pitch, knee, and ankle_pitch have non-zero defaults
DEFAULT_JOINT_POS = np.array([
    -0.1,   # left_hip_pitch_joint (.*_hip_pitch_joint)
    0.0,    # left_hip_roll_joint
    0.0,    # left_hip_yaw_joint
    0.3,    # left_knee_joint (.*_knee_joint)
    -0.2,   # left_ankle_pitch_joint (.*_ankle_pitch_joint)
    0.0,    # left_ankle_roll_joint
    -0.1,   # right_hip_pitch_joint (.*_hip_pitch_joint)
    0.0,    # right_hip_roll_joint
    0.0,    # right_hip_yaw_joint
    0.3,    # right_knee_joint (.*_knee_joint)
    -0.2,   # right_ankle_pitch_joint (.*_ankle_pitch_joint)
    0.0,    # right_ankle_roll_joint
    0.0,    # waist_yaw_joint
    0.0,    # waist_roll_joint
    0.0,    # waist_pitch_joint
    0.0,    # left_shoulder_pitch_joint
    0.0,    # left_shoulder_roll_joint
    0.0,    # left_shoulder_yaw_joint
    0.0,    # left_elbow_joint
    0.0,    # left_wrist_roll_joint
    0.0,    # left_wrist_pitch_joint
    0.0,    # left_wrist_yaw_joint
    0.0,    # right_shoulder_pitch_joint
    0.0,    # right_shoulder_roll_joint
    0.0,    # right_shoulder_yaw_joint
    0.0,    # right_elbow_joint
    0.0,    # right_wrist_roll_joint
    0.0,    # right_wrist_pitch_joint
    0.0,    # right_wrist_yaw_joint
], dtype=np.float32)

# Global action rescale factor
ACTION_RESCALE = 5.0
