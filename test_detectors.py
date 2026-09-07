
import cv2
import time
import mediapipe as mp

from detectors.ear import get_ear
from detectors.mar import get_mar
from detectors.head_pose import get_head_pose
from detectors.calibration import calibrate


# ============================================================
# MediaPipe Face Landmarker setup
# ============================================================

MODEL_PATH = "models/face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

face_landmarker = FaceLandmarker.create_from_options(options)


# ============================================================
# Open webcam
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    face_landmarker.close()
    exit()


# ============================================================
# Calibration
# ============================================================

baseline = calibrate(
    cap,
    face_landmarker,
    duration_seconds=5
)

# Thresholds relative to this person's calibrated baseline
EAR_DROWSY_THRESHOLD = baseline["ear_baseline"] * 0.75   # eyes ~25% more closed than normal
MAR_YAWN_THRESHOLD = baseline["mar_baseline"] * 1.8       # mouth ~80% more open than normal

EAR_CONSEC_FRAMES = 20   # tune this based on your webcam's FPS
MAR_CONSEC_FRAMES = 15

ear_counter = 0
mar_counter = 0

# ============================================================
# Main detection loop
# ============================================================

frame_timestamp_ms = baseline["last_timestamp_ms"]

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        continue

    frame_h, frame_w = frame.shape[:2]

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    frame_timestamp_ms += 1

    results = face_landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )


    # ========================================================
    # Face detected
    # ========================================================

    if results.face_landmarks:

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

        pitch, yaw, roll = get_head_pose(
            landmarks,
            frame_w,
            frame_h
        )


        # ====================================================
        # Personalized eye closure detection
        # ====================================================

        if ear < EAR_DROWSY_THRESHOLD:
            ear_counter += 1
        else:
            ear_counter = 0


        # ====================================================
        # Personalized yawn detection
        # ====================================================

        if mar > MAR_YAWN_THRESHOLD:
            mar_counter += 1
        else:
            mar_counter = 0


        # ====================================================
        # Testing alerts
        # ====================================================

        if ear_counter >= EAR_CONSEC_FRAMES:

            cv2.putText(
                frame,
                'EYES CLOSED - DROWSY!',
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )


        if mar_counter >= MAR_CONSEC_FRAMES:

            cv2.putText(
                frame,
                'YAWN DETECTED!',
                (30, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )


        # ====================================================
        # Display normal measurements
        # ====================================================

        cv2.putText(
            frame,
            f'EAR: {ear:.2f} '
            f'(base {baseline["ear_baseline"]:.2f})',
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f'MAR: {mar:.2f} '
            f'(base {baseline["mar_baseline"]:.2f})',
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f'Pitch: {pitch:.1f}  '
            f'Yaw: {yaw:.1f}  '
            f'Roll: {roll:.1f}',
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ========================================================
    # Show webcam
    # ========================================================

    cv2.imshow(
        'Detector Test - Press Q to quit',
        frame
    )

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break


# ============================================================
# Cleanup
# ============================================================

cap.release()
face_landmarker.close()
cv2.destroyAllWindows()

