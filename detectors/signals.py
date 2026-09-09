
from detectors.ear import get_ear
from detectors.mar import get_mar
from detectors.head_pose import get_head_pose


def get_frame_signals(landmarks, frame_w, frame_h, baseline):

    ear = get_ear(
        landmarks,
        frame_w,
        frame_h
    )

    mar = get_mar(
        landmarks,
        frame_w,
        frame_h
    )

    pitch, yaw, roll = get_head_pose(
        landmarks,
        frame_w,
        frame_h
    )

    return {
        "ear": ear,
        "mar": mar,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "ear_baseline": baseline["ear_baseline"],
        "mar_baseline": baseline["mar_baseline"]
    }

