
import time


class AlertManager:
    """
    Tracks fatigue score over time and decides the alert tier.

    Tiers:
        0 = Normal
        1 = Mild fatigue
        2 = Serious warning
        3 = Critical alarm

    The score must remain elevated for a sustained period
    before an alert is triggered.
    """

    def __init__(self, sustained_seconds=1.5):

        self.sustained_seconds = sustained_seconds

        self.high_score_start_time = None

        self.current_tier = 0


    def update(self, score):

        # Determine which alert tier the score belongs to.
        if score >= 85:
            threshold_tier = 3

        elif score >= 60:
            threshold_tier = 2

        elif score >= 30:
            threshold_tier = 1

        else:
            threshold_tier = 0


        # ----------------------------------------------------
        # No fatigue detected
        # ----------------------------------------------------

        if threshold_tier == 0:

            self.high_score_start_time = None
            self.current_tier = 0

            return self.current_tier


        # ----------------------------------------------------
        # Fatigue detected
        # ----------------------------------------------------

        if self.high_score_start_time is None:

            self.high_score_start_time = time.time()


        elapsed = (
            time.time() -
            self.high_score_start_time
        )


        # ----------------------------------------------------
        # Require sustained elevated score
        # ----------------------------------------------------

        if elapsed >= self.sustained_seconds:

            self.current_tier = threshold_tier

        else:

            self.current_tier = 0


        return self.current_tier


    def get_alert_text_and_color(self, tier):

        if tier == 0:

            return None, None


        elif tier == 1:

            return (
                "MILD FATIGUE - STAY ALERT",
                (0, 255, 255)
            )


        elif tier == 2:

            return (
                "WARNING: DROWSINESS DETECTED",
                (0, 165, 255)
            )


        elif tier == 3:

            return (
                "CRITICAL: WAKE UP!",
                (0, 0, 255)
            )


        return None, None


def play_alert_sound(tier):

    try:

        import winsound

        if tier == 3:

            winsound.Beep(
                1000,
                400
            )

        elif tier == 2:

            winsound.Beep(
                750,
                250
            )

        elif tier == 1:

            winsound.Beep(
                500,
                150
            )

    except ImportError:

        print("\a")

