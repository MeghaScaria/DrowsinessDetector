
import cv2
import time
import mediapipe as mp

from detectors.ear import get_ear
from detectors.mar import get_mar
frame_timestamp_ms = 0

def calibrate(cap, face_landmarker, duration_seconds=5):
    """
    Runs for `duration_seconds`, asks the user to sit normally with eyes open,
    mouth closed, and records average EAR/MAR as personal baseline.

    Returns:
        {
            "ear_baseline": float,
            "mar_baseline": float
        }
    """

    ear_readings = []
    mar_readings = []

    start_time = time.time()

    # Timestamp used by MediaPipe VIDEO mode.
    frame_timestamp_ms = 0

    print(
        f"Calibrating... look at the camera normally "
        f"for {duration_seconds} seconds."
    )

    while time.time() - start_time < duration_seconds:

        success, frame = cap.read()

        if not success:
            continue

        frame_h, frame_w = frame.shape[:2]

        # OpenCV uses BGR; MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert NumPy/OpenCV image to MediaPipe Image.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # MediaPipe VIDEO mode requires increasing timestamps.
        frame_timestamp_ms += 1

        results = face_landmarker.detect_for_video(
            mp_image,
            frame_timestamp_ms
        )

        # New Face Landmarker API:
        # results.face_landmarks
        if results.face_landmarks:

            # First detected face.
            landmarks = results.face_landmarks[0]

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

            ear_readings.append(ear)
            mar_readings.append(mar)

        remaining = max(
            0,
            int(duration_seconds - (time.time() - start_time))
        )

        cv2.putText(
            frame,
            f'Calibrating... {remaining}s',
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            'Calibration',
            frame
        )

        cv2.waitKey(1)

    cv2.destroyWindow('Calibration')

    # Use defaults if no face was detected during calibration.
    ear_baseline = (
        sum(ear_readings) / len(ear_readings)
        if ear_readings
        else 0.25
    )

    mar_baseline = (
        sum(mar_readings) / len(mar_readings)
        if mar_readings
        else 0.4
    )

    print(
        f"Calibration done. "
        f"EAR baseline: {ear_baseline:.3f}, "
        f"MAR baseline: {mar_baseline:.3f}"
    )

    return {
    "ear_baseline": ear_baseline,
    "mar_baseline": mar_baseline,
    "last_timestamp_ms": frame_timestamp_ms
}

