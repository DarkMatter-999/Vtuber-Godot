"""
Modified from https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py
"""
import cv2
import numpy as np

class PoseEstimator:

    def __init__(self, img_size=(480, 640)):
        self.size = img_size

        self.model_points_full = self.get_full_model_points()

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = None
        self.t_vec = None

    def get_full_model_points(self, filename='model.txt'): #model.txt from https://github.com/mmmmmm44/VTuber-Python-Unity/blob/main/model.txt
        """Get all 468 3D model points from file"""
        raw_value = []

        with open(filename) as file:
            for line in file:
                raw_value.append(line)

        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (-1, 3))

        return model_points

    def solve_pose_by_all_points(self, image_points):
        """
        Solve pose from all the 468 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_full, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_full,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def draw_axes(self, img, R, t):
            img	= cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeefs, R, t, 20)
