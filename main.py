import cv2
import mediapipe as mp
import numpy as np

from face_mesh import FaceMeshDetector

from facial_features import FacialFeatures, Eyes

from pose_detection import PoseEstimator

from motion_stabilizer import Stabilizer

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    face_detector = FaceMeshDetector()

    tmp, img = cap.read()

    pose_estimator = PoseEstimator((img.shape[0], img.shape[1]))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    # for eyes
    eyes_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    # for mouth_dist
    mouth_dist_stabilizer = Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1
    )

    while cap.isOpened():
        tmp, img = cap.read()

        if not tmp:
            continue

        image_points = np.zeros((pose_estimator.model_points_full.shape[0], 2))   #468 x2

        img_facemesh, faces = face_detector.getMesh(img)

        img = cv2.flip(img, 1)

        if faces:
            faces = faces[0]

            for i in range(len(image_points)):
                image_points[i, 0] = faces[i][0]
                image_points[i, 1] = faces[i][1]

            pose = pose_estimator.solve_pose_by_all_points(image_points)

            x_left, y_left, x_ratio_left, y_ratio_left = FacialFeatures.detect_iris(img, faces, Eyes.LEFT)
            x_right, y_right, x_ratio_right, y_ratio_right = FacialFeatures.detect_iris(img, faces, Eyes.RIGHT)


            ear_left = FacialFeatures.eye_aspect_ratio(image_points, Eyes.LEFT)
            ear_right = FacialFeatures.eye_aspect_ratio(image_points, Eyes.RIGHT)

            pose_eye = [ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right]

            mar = FacialFeatures.mouth_aspect_ratio(image_points)
            mouth_distance = FacialFeatures.mouth_distance(image_points)

            # Stabilize the pose.
            steady_pose = []
            pose_np = np.array(pose).flatten()

            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                steady_pose.append(ps_stb.state[0])

            steady_pose = np.reshape(steady_pose, (-1, 3))

            # stabilize the eyes value
            steady_pose_eye = []
            for value, ps_stb in zip(pose_eye, eyes_stabilizers):
                ps_stb.update([value])
                steady_pose_eye.append(ps_stb.state[0])

            mouth_dist_stabilizer.update([mouth_distance])
            steady_mouth_dist = mouth_dist_stabilizer.state[0]

            roll = np.clip(np.degrees(steady_pose[0][1]), -90, 90)
            pitch = np.clip(-(180 + np.degrees(steady_pose[0][0])), -90, 90)
            yaw =  np.clip(np.degrees(steady_pose[0][2]), -90, 90)

            # print(steady_pose)
            pose_estimator.draw_axes(img_facemesh, steady_pose[0], steady_pose[1])

        cv2.imshow("Display", img_facemesh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
