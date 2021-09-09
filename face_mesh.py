
import cv2
import mediapipe as mp

class FaceMeshDetector:
    def __init__(self,
                max_face = 1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5):

        self.max_face = max_face
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        self.face_mesh = self.mp_face_mesh.FaceMesh(
                                                False,
                                                self.max_face,
                                                self.min_detection_confidence,
                                                self.min_tracking_confidence)

        self.face = []

    def getMesh(self, image):


        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.imgH, self.imgW, self.imgC = image.shape

        self.face = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.drawing_spec,
                    connection_drawing_spec=self.drawing_spec)

                face = []
                for id, landmark in enumerate(face_landmarks.landmark):
                    x, y = int(landmark.x * self.imgW), int(landmark.y * self.imgH)
                    face.append([x, y])


            self.face.append(face)

        return image, self.face
