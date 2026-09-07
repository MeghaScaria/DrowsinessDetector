from scipy.spatial import distance as dist

# MediaPipe indices: inner lip top/bottom, and mouth corners
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_CORNER = 78
RIGHT_CORNER = 308

def _landmark_to_pixel(landmark, frame_w, frame_h):
    return (int(landmark.x * frame_w), int(landmark.y * frame_h))

def get_mar(landmarks, frame_w, frame_h):
    """
    Returns Mouth Aspect Ratio (float). Higher value = mouth more open (yawning).
    Typical closed-mouth range: 0.3-0.5, yawning usually pushes above 0.7-0.8
    (you WILL need to tune this by watching your own values while yawning vs talking normally).
    """
    top = _landmark_to_pixel(landmarks[UPPER_LIP], frame_w, frame_h)
    bottom = _landmark_to_pixel(landmarks[LOWER_LIP], frame_w, frame_h)
    left = _landmark_to_pixel(landmarks[LEFT_CORNER], frame_w, frame_h)
    right = _landmark_to_pixel(landmarks[RIGHT_CORNER], frame_w, frame_h)

    vertical = dist.euclidean(top, bottom)
    horizontal = dist.euclidean(left, right)

    return vertical / horizontal