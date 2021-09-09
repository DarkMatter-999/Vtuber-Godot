import cv2
import mediapipe as mp
import numpy as np

from face_mesh import FaceMeshDetector
# from vtube.facial_landmark import FaceMeshDetector

from facial_features import FacialFeatures, Eyes



def main():
    cap = cv2.VideoCapture(0)

    face_detector = FaceMeshDetector()

    while cap.isOpened():
        tmp, img = cap.read()

        if not tmp:
            continue

        image_points = np.zeros((468, 2))   #468

        img_facemesh, faces = face_detector.getMesh(img)

        img = cv2.flip(img, 1)

        if faces:
            faces = faces[0]

            for i in range(len(image_points)):
                image_points[i, 0] = faces[i][0]
                image_points[i, 1] = faces[i][1]

            x_left, y_left, x_ratio_left, y_ratio_left = FacialFeatures.detect_iris(img, faces, Eyes.LEFT)
            x_right, y_right, x_ratio_right, y_ratio_right = FacialFeatures.detect_iris(img, faces, Eyes.RIGHT)


            ear_left = FacialFeatures.eye_aspect_ratio(image_points, Eyes.LEFT)
            ear_right = FacialFeatures.eye_aspect_ratio(image_points, Eyes.RIGHT)

            pose_eye = [ear_left, ear_right, x_ratio_left, y_ratio_left, x_ratio_right, y_ratio_right]

            mar = FacialFeatures.mouth_aspect_ratio(image_points)
            mouth_distance = FacialFeatures.mouth_distance(image_points)

        cv2.imshow("Display", img_facemesh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
