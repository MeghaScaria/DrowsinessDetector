from scipy.spatial import distance as dist


# MediaPipe mouth landmarks
LEFT_CORNER = 61
RIGHT_CORNER = 291

# Vertical mouth pairs
UPPER_LOWER_1 = (13, 14)
UPPER_LOWER_2 = (82, 87)
UPPER_LOWER_3 = (312, 317)


def _landmark_to_pixel(landmark, frame_w, frame_h):
    return (
        int(landmark.x * frame_w),
        int(landmark.y * frame_h)
    )


def get_mar(landmarks, frame_w, frame_h):
    """
    Calculates Mouth Aspect Ratio (MAR).

    Uses three vertical mouth measurements averaged together
    and normalizes them by the mouth width.

    Higher MAR = more mouth opening.
    """

    left = _landmark_to_pixel(
        landmarks[LEFT_CORNER],
        frame_w,
        frame_h
    )

    right = _landmark_to_pixel(
        landmarks[RIGHT_CORNER],
        frame_w,
        frame_h
    )

    horizontal = dist.euclidean(left, right)

    vertical_distances = []

    for upper_idx, lower_idx in [
        UPPER_LOWER_1,
        UPPER_LOWER_2,
        UPPER_LOWER_3
    ]:

        upper = _landmark_to_pixel(
            landmarks[upper_idx],
            frame_w,
            frame_h
        )

        lower = _landmark_to_pixel(
            landmarks[lower_idx],
            frame_w,
            frame_h
        )

        vertical_distances.append(
            dist.euclidean(upper, lower)
        )

    average_vertical = sum(vertical_distances) / len(
        vertical_distances
    )

    return average_vertical / horizontal