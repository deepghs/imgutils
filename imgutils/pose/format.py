from enum import IntEnum, unique

import numpy as np

OP18_BODY_MIN = 0
OP18_BODY_MAX = 17
OP18_LEFT_FOOT_MIN = 18
OP18_LEFT_FOOT_MAX = 20
OP18_RIGHT_FOOT_MIN = 21
OP18_RIGHT_FOOT_MAX = 23
OP18_FACE_MIN = 24
OP18_FACE_MAX = 91
OP18_LEFT_HAND_MIN = 92
OP18_LEFT_HAND_MAX = 112
OP18_RIGHT_HAND_MIN = 113
OP18_RIGHT_HAND_MAX = 135


@unique
class OpenPose18(IntEnum):
    """
    Enumeration class representing the OpenPose 18 keypoint indices.

    The enumeration provides symbolic names for the keypoint indices, making it more readable and maintainable
    when accessing specific keypoints in the OP18 keypoint set.

    The keypoint indices are categorized into different body parts such as nose, neck, shoulders, elbows, wrists,
    hips, knees, ankles, eyes, ears, feet, and hands.
    """
    NOSE = 0
    NECK = 1
    RIGHT_SHOULDER = 2
    RIGHT_ELBOW = 3
    RIGHT_WRIST = 4
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 6
    LEFT_WRIST = 7
    RIGHT_HIP = 8
    RIGHT_KNEE = 9
    RIGHT_ANKLE = 10
    LEFT_HIP = 11
    LEFT_KNEE = 12
    LEFT_ANKLE = 13
    RIGHT_EYE = 14
    LEFT_EYE = 15
    RIGHT_EAR = 16
    LEFT_EAR = 17

    LEFT_BIG_TOE = 18
    LEFT_SMALL_TOE = 19
    LEFT_HEEL = 20
    RIGHT_BIG_TOE = 21
    RIGHT_SMALL_TOE = 22
    RIGHT_HEEL = 23


class OP18KeyPointSet:
    """
    Class representing a set of keypoints detected by the OpenPose 18 (OP18) model.

    This class provides convenient properties to access keypoints for different body parts, including the body,
    left foot, right foot, face, left hand, and right hand.

    :param all_: NumPy array containing the coordinates and confidence scores of all keypoints.
    :type all_: np.ndarray
    """

    def __init__(self, all_: np.ndarray):
        self.all: np.ndarray = all_

    @property
    def body(self):
        """Property representing the keypoints for the body."""
        return self.all[OP18_BODY_MIN:OP18_BODY_MAX + 1]

    @property
    def left_foot(self):
        """Property representing the keypoints for the left foot."""
        return self.all[OP18_LEFT_FOOT_MIN:OP18_LEFT_FOOT_MAX + 1]

    @property
    def right_foot(self):
        """Property representing the keypoints for the right foot."""
        return self.all[OP18_RIGHT_FOOT_MIN:OP18_RIGHT_FOOT_MAX + 1]

    @property
    def face(self):
        """Property representing the keypoints for the face."""
        return self.all[OP18_FACE_MIN:OP18_FACE_MAX + 1]

    @property
    def left_hand(self):
        """Property representing the keypoints for the left hand."""
        return self.all[OP18_LEFT_HAND_MIN:OP18_LEFT_HAND_MAX + 1]

    @property
    def right_hand(self):
        """Property representing the keypoints for the right hand."""
        return self.all[OP18_RIGHT_HAND_MIN:OP18_RIGHT_HAND_MAX + 1]

    def __mul__(self, multiplier):
        """
        Multiply the coordinates of all keypoints by a scalar multiplier.

        :param multiplier: The scalar multiplier.
        :type multiplier: Union[float, int]

        :return: New OP18KeyPointSet with scaled coordinates.
        :rtype: OP18KeyPointSet

        :raises TypeError: If the type of the multiplier is not float or int.
        """
        if isinstance(multiplier, (float, int)):
            new_all = self.all.copy()
            new_all[:, 0] *= multiplier
            new_all[:, 1] *= multiplier
            return OP18KeyPointSet(new_all)
        else:
            raise TypeError(f'Invalid type of multiplier - {multiplier!r}')

    def __truediv__(self, divisor):
        """
        Divide the coordinates of all keypoints by a scalar divisor.

        :param divisor: The scalar divisor.
        :type divisor: Union[float, int]

        :return: New OP18KeyPointSet with scaled coordinates.
        :rtype: OP18KeyPointSet

        :raises TypeError: If the type of the divisor is not float or int.
        """
        if isinstance(divisor, (float, int)):
            return self * (1.0 / divisor)
        else:
            raise TypeError(f'Invalid type of divisor - {divisor!r}.')
