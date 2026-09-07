import cv2
import numpy as np

# 2D landmark indices we'll use for pose estimation
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 263
RIGHT_EYE_CORNER = 33
LEFT_MOUTH_CORNER = 291
RIGHT_MOUTH_CORNER = 61

POSE_LANDMARK_IDS = [NOSE_TIP, CHIN, LEFT_EYE_CORNER, RIGHT_EYE_CORNER, LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER]

# Generic 3D face model points (in mm, approximate average face geometry)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float64)

def get_head_pose(landmarks, frame_w, frame_h):
    """
    Returns (pitch, yaw, roll) in degrees.
    pitch: nodding up/down (negative = looking down, drowsy nodding)
    yaw: turning left/right (looking away from road)
    roll: tilting sideways
    """
    image_points = np.array([
        (landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in POSE_LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs
    )

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = euler_angles.flatten()
    return pitch, yaw, roll