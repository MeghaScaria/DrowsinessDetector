import cv2
import time
from detectors.ear import get_ear
from detectors.mar import get_mar

def calibrate(cap, face_mesh, mp_face_mesh, duration_seconds=5):
    """
    Runs for `duration_seconds`, asks the user to sit normally with eyes open,
    mouth closed, and records average EAR/MAR as personal baseline.
    Returns a dict: {"ear_baseline": float, "mar_baseline": float}
    """
    ear_readings = []
    mar_readings = []
    start_time = time.time()

    print(f"Calibrating... look at the camera normally for {duration_seconds} seconds.")

    while time.time() - start_time < duration_seconds:
        success, frame = cap.read()
        if not success:
            continue

        frame_h, frame_w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            ear_readings.append(get_ear(landmarks, frame_w, frame_h))
            mar_readings.append(get_mar(landmarks, frame_w, frame_h))

        remaining = int(duration_seconds - (time.time() - start_time))
        cv2.putText(frame, f'Calibrating... {remaining}s', (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow('Calibration', frame)
        cv2.waitKey(1)

    cv2.destroyWindow('Calibration')

    ear_baseline = sum(ear_readings) / len(ear_readings) if ear_readings else 0.25
    mar_baseline = sum(mar_readings) / len(mar_readings) if mar_readings else 0.4

    print(f"Calibration done. EAR baseline: {ear_baseline:.3f}, MAR baseline: {mar_baseline:.3f}")
    return {"ear_baseline": ear_baseline, "mar_baseline": mar_baseline}