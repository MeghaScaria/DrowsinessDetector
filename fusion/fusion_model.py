
def compute_fatigue_score(signals, thresholds, weights=None):
    """
    Combines EAR, MAR, head pose and yaw into a single
    fatigue/distraction score from 0-100.

    Higher score = stronger evidence of fatigue/distraction.

    Each individual signal is normalized against a threshold
    before being combined.
    """

    if weights is None:
        weights = {
            "eye": 0.40,
            "yawn": 0.25,
            "head_nod": 0.25,
            "head_turn": 0.10
        }

    ear = signals["ear"]
    mar = signals["mar"]
    pitch = signals["pitch"]
    yaw = signals["yaw"]

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    ear_threshold = thresholds["ear"]
    mar_threshold = thresholds["mar"]
    pitch_threshold = thresholds["pitch"]
    yaw_threshold = thresholds["yaw"]

    # --------------------------------------------------------
    # EAR risk
    #
    # 0 = normal
    # 1 = significantly below drowsiness threshold
    # --------------------------------------------------------

    if ear >= ear_threshold:
        eye_risk = 0.0
    else:
        eye_risk = min(
            1.0,
            (ear_threshold - ear) / ear_threshold
        )

    # --------------------------------------------------------
    # MAR / yawn risk
    #
    # 0 = normal
    # 1 = clearly above yawn threshold
    # --------------------------------------------------------

    if mar <= mar_threshold:
        yawn_risk = 0.0
    else:
        yawn_risk = min(
            1.0,
            (mar - mar_threshold) / mar_threshold
        )

    # --------------------------------------------------------
    # Head nod risk
    #
    # IMPORTANT:
    # The direction depends on your pose-estimation setup.
    # This assumes lower pitch = more downward head movement.
    # We will tune this after observing your values.
    # --------------------------------------------------------

    if pitch >= pitch_threshold:
        head_nod_risk = 0.0
    else:
        head_nod_risk = min(
            1.0,
            (pitch_threshold - pitch) / 20.0
        )

    # --------------------------------------------------------
    # Head turn / distraction risk
    # --------------------------------------------------------

    if abs(yaw) <= yaw_threshold:
        head_turn_risk = 0.0
    else:
        head_turn_risk = min(
            1.0,
            (abs(yaw) - yaw_threshold) / 30.0
        )

    # --------------------------------------------------------
    # Weighted fusion
    # --------------------------------------------------------

    score = (
        weights["eye"] * eye_risk +
        weights["yawn"] * yawn_risk +
        weights["head_nod"] * head_nod_risk +
        weights["head_turn"] * head_turn_risk
    ) * 100

    return {
    "score": round(score, 1),
    "eye_risk": round(eye_risk, 2),
    "yawn_risk": round(yawn_risk, 2),
    "head_nod_risk": round(head_nod_risk, 2),
    "head_turn_risk": round(head_turn_risk, 2)
    }

