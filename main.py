
import cv2
import time
import mediapipe as mp

from detectors.signals import get_frame_signals
from detectors.calibration import calibrate

from fusion.fusion_model import compute_fatigue_score
from fusion.alerts import AlertManager, play_alert_sound


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/face_landmarker.task"

# ------------------------------------------------------------
# Personalized EAR threshold
# ------------------------------------------------------------

# Eyes are considered unusually closed when EAR drops
# below 75% of the person's calibrated normal EAR.

# ------------------------------------------------------------
# MAR threshold
# ------------------------------------------------------------

# Temporary tuned threshold based on observed MAR values.
# We are NOT using baseline * 1.8 because your calibrated
# MAR baseline was unrealistically small.
MAR_YAWN_THRESHOLD = 0.45

# ------------------------------------------------------------
# Head pose thresholds
# ------------------------------------------------------------

# IMPORTANT:
# Your current head-pose implementation has a pitch-direction
# issue that we still need to tune using your actual values.
#
# This assumes that lower pitch means downward head movement.
# If your setup produces the opposite sign, flip the comparison
# inside fusion/fusion_model.py.
PITCH_DROWSY_THRESHOLD = -25

# Looking away from the road/camera.
YAW_DISTRACTION_THRESHOLD = 25


# ============================================================
# MediaPipe Face Landmarker setup
# ============================================================

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


face_landmarker = FaceLandmarker.create_from_options(
    options
)


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


# ============================================================
# Personalized thresholds
# ============================================================

EAR_DROWSY_THRESHOLD = (
    baseline["ear_baseline"] * 0.75
)


thresholds = {
    "ear": EAR_DROWSY_THRESHOLD,
    "mar": MAR_YAWN_THRESHOLD,
    "pitch": PITCH_DROWSY_THRESHOLD,
    "yaw": YAW_DISTRACTION_THRESHOLD
}


# ============================================================
# Print calibration information
# ============================================================

print()
print("========================================")
print("Calibration Results")
print("========================================")
print(
    f"EAR baseline:       "
    f"{baseline['ear_baseline']:.3f}"
)
print(
    f"MAR baseline:       "
    f"{baseline['mar_baseline']:.3f}"
)
print(
    f"EAR threshold:      "
    f"{EAR_DROWSY_THRESHOLD:.3f}"
)
print(
    f"MAR yawn threshold: "
    f"{MAR_YAWN_THRESHOLD:.3f}"
)
print(
    f"Pitch threshold:    "
    f"{PITCH_DROWSY_THRESHOLD:.1f}"
)
print(
    f"Yaw threshold:      "
    f"{YAW_DISTRACTION_THRESHOLD:.1f}"
)
print("========================================")
print()


# ============================================================
# Timestamp
# ============================================================

# Calibration has already used the FaceLandmarker object.
# Continue from its last timestamp so detect_for_video()
# always receives increasing timestamps.
frame_timestamp_ms = baseline["last_timestamp_ms"]


# ============================================================
# Fusion + alert system
# ============================================================

alert_manager = AlertManager(
    sustained_seconds=0.5
)


# Prevent the alert sound from playing every frame.
last_sound_time = 0.0
SOUND_COOLDOWN = 2.0


# ============================================================
# Main detection loop
# ============================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        continue


    # --------------------------------------------------------
    # Frame dimensions
    # --------------------------------------------------------

    frame_h, frame_w = frame.shape[:2]


    # --------------------------------------------------------
    # OpenCV BGR → RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Convert to MediaPipe Image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # --------------------------------------------------------
    # Increment timestamp
    # --------------------------------------------------------

    frame_timestamp_ms += 1


    # --------------------------------------------------------
    # Run Face Landmarker
    # --------------------------------------------------------

    results = face_landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )


    # ========================================================
    # Face detected
    # ========================================================

    if results.face_landmarks:

        # First detected face
        landmarks = results.face_landmarks[0]


        # ====================================================
        # Calculate all frame signals
        # ====================================================

        signals = get_frame_signals(
            landmarks,
            frame_w,
            frame_h,
            baseline
        )


        # ====================================================
        # Compute fatigue score
        # ====================================================

        fusion = compute_fatigue_score(
        signals,
        thresholds
        )


        # ====================================================
        # Update alert state
        # ====================================================
        score = fusion["score"]

        eye_risk = fusion["eye_risk"]
        yawn_risk = fusion["yawn_risk"]
        head_nod_risk = fusion["head_nod_risk"]
        head_turn_risk = fusion["head_turn_risk"]
        
        tier = alert_manager.update(score)


        # ====================================================
        # Display EAR and MAR
        # ====================================================

        cv2.putText(
            frame,
            f'EAR: {signals["ear"]:.2f}  '
            f'MAR: {signals["mar"]:.2f}',
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ====================================================
        # Display head pose
        # ====================================================

        cv2.putText(
            frame,
            f'Pitch: {signals["pitch"]:.1f}  '
            f'Yaw: {signals["yaw"]:.1f}  '
            f'Roll: {signals["roll"]:.1f}',
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


        # ====================================================
        # Display thresholds
        # ====================================================

        cv2.putText(
            frame,
            f'EAR Thresh: {EAR_DROWSY_THRESHOLD:.2f}  '
            f'MAR Thresh: {MAR_YAWN_THRESHOLD:.2f}',
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )


        # ====================================================
        # Display fatigue score
        # ====================================================

        cv2.putText(
            frame,
            f'Fatigue Score: {score}/100',
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # ========================================================
        # Display individual fusion risks
        # ========================================================

        cv2.putText(
            frame,
            f'Eye: {eye_risk:.2f}  '
            f'Yawn: {yawn_risk:.2f}',
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f'Nod: {head_nod_risk:.2f}  '
            f'Turn: {head_turn_risk:.2f}',
            (30, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # Display alert
        # ====================================================

        alert_text, color = (
            alert_manager.get_alert_text_and_color(tier)
        )


        if alert_text:

            cv2.putText(
                frame,
                alert_text,
                (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                3
            )


            # ------------------------------------------------
            # Play alert sound with cooldown
            # ------------------------------------------------

            now = time.time()

            if now - last_sound_time > SOUND_COOLDOWN:

                play_alert_sound(tier)

                last_sound_time = now


    # ========================================================
    # No face detected
    # ========================================================

    else:

        cv2.putText(
            frame,
            'No face detected',
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    # ========================================================
    # Display webcam
    # ========================================================

    cv2.imshow(
        'Driver Drowsiness Detector - Press Q to quit',
        frame
    )


    # ========================================================
    # Quit
    # ========================================================

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break


# ============================================================
# Cleanup
# ============================================================

cap.release()
face_landmarker.close()
cv2.destroyAllWindows()

