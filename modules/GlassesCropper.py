import math

import cv2
import numpy as np


class GlassesCroper():
    def __init__(self, idx_left=2, idx_right=14, scale_radius=1):
        self.idx_left = idx_left  # The lowest point on left of the glasses
        self.idx_right = idx_right  # The lowest point on right of the glasses
        self.scale_radius = scale_radius
        self.eye_constant_points = [26, 25, 24, 23, 22, 21, 20, 19, 18, 17]

    def run(self, landmarks, pose, color=(0, 0, 0), face=None):
        """
        :param landmarks: 3DDFA's landmarks
        :typ3 landmarks: list
        :param pose: [yawl_angle, pitch_angle, rotate_angle]
        :type pose: tuple
        :param color: mask's color, default = (0, 0, 0)
        :type color: tuple
        :param face: FaceVerify's face object
        :return: draw_mask_status, mask_face
        :rtype: bool, img
        """

        glasses_face = face.copy()

        try:
            cal_glassespoints_status, glasses_points = self.calculate_glassses_points(landmarks, pose)
            if not cal_glassespoints_status:
                return cal_glassespoints_status, glasses_face
            else:
                # Calculate face frame's mask point
                mask_points = np.array(glasses_points)

                cv2.fillPoly(glasses_face, np.int32(np.array([glasses_points])), color=color, lineType=cv2.LINE_AA)
                return cal_glassespoints_status, glasses_face
        except Exception as ex:
            print(ex)
            return False, glasses_face

    def calculate_glassses_points(self, landmarks, pose):
        mask_points = []

        # if headpose is not frontal, draw mask with round arc
        center_2 = landmarks[self.idx_left]
        center_1 = landmarks[self.idx_right]
        radius_2 = int(
            np.sqrt((landmarks[self.idx_left][0] - landmarks[26][0]) ** 2 +
                    (landmarks[self.idx_left][1] - landmarks[26][1]) ** 2) * self.scale_radius)
        radius_1 = int(
            np.sqrt((landmarks[self.idx_right][0] - landmarks[17][0]) ** 2 +
                    (landmarks[self.idx_right][1] - landmarks[17][1]) ** 2) * self.scale_radius)

        # Bias angle: if face is big, need longer round arc to cover mask area
        bias_angle_1 = int(np.sqrt(radius_1) - pose[1] / 2)
        bias_angle_2 = -int(np.sqrt(radius_2) - pose[1] / 2)

        start_angle_1 = np.arcsin((landmarks[26][1] - landmarks[self.idx_left][1]) / radius_1)
        start_angle_1 = -int(start_angle_1 * 180 / np.pi) if not math.isnan(start_angle_1) else start_angle_1
        start_angle_2 = np.arcsin((landmarks[17][1] - landmarks[self.idx_right][1]) / radius_2)
        start_angle_2 = int(start_angle_2 * 180 / np.pi) if not math.isnan(start_angle_2) else start_angle_2

        # check divide by zero
        if math.isnan(start_angle_1) or math.isnan(start_angle_2) or radius_1 == 0 or radius_2 == 0:
            return False, []

        if pose[0] > -10:
            for idx in range(self.idx_right, 16):
                point = landmarks[idx]
                mask_points.append(point)
        else:
            for angle in range(45 + start_angle_2 + bias_angle_2, start_angle_2, -5):
                point = (center_2[0] + int(np.cos(angle * np.pi / 180) * radius_2),
                         center_2[1] + int(np.sin(angle * np.pi / 180) * radius_2))
                mask_points.append(point)

        for idx in self.eye_constant_points:
            mask_points.append(landmarks[idx])

        if pose[0] < 10:
            for idx in range(0, self.idx_left + 1):
                point = landmarks[idx]
                mask_points.append(point)
        else:
            for angle in range(180 + start_angle_1, 135 + start_angle_1 + bias_angle_1, -5):
                point = (center_1[0] + int(np.cos(angle * np.pi / 180) * radius_1),
                         center_1[1] + int(np.sin(angle * np.pi / 180) * radius_1))
                mask_points.append(point)

        mask_points.append(landmarks[28])

        return True, mask_points
