from scipy.spatial import distance as dist

# MediaPipe 468-landmark indices for the 6 EAR points per eye
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def _landmark_to_pixel(landmark, frame_w, frame_h):
    return (int(landmark.x * frame_w), int(landmark.y * frame_h))

def get_ear(landmarks, frame_w, frame_h):
    """
    Returns the average Eye Aspect Ratio (float) across both eyes.
    Lower value = more closed eyes. Typical open-eye range: 0.25-0.35
    """
    def ear_for_eye(eye_points):
        coords = [_landmark_to_pixel(landmarks[i], frame_w, frame_h) for i in eye_points]
        A = dist.euclidean(coords[1], coords[5])
        B = dist.euclidean(coords[2], coords[4])
        C = dist.euclidean(coords[0], coords[3])
        return (A + B) / (2.0 * C)

    left_ear = ear_for_eye(LEFT_EYE)
    right_ear = ear_for_eye(RIGHT_EYE)
    return (left_ear + right_ear) / 2.0