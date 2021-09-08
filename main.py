import cv2
import mediapipe as mp
import numpy as np

from face_mesh import FaceMeshDetector


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        tmp, img = cap.read()

        cv2.imshow("Display", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
