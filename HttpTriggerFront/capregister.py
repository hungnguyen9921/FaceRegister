import numpy as np
import math
import cv2
import collections


class CaptureRegisterFace:
    def __init__(self, left=-25, right=16, middle_range=35):
        self.left = left
        self.right = right
        self.mid = middle_range
        self.state = "LEFT"

    def checkleft(self, frame, image_points):
        rotation_angle = self.detectHeadpose(frame, image_points)[1]
        print(rotation_angle)
        if (rotation_angle < self.left):
            return None, "REGISTER_DONE_LEFT"
        else:
            return None, None
        return None, None

    def checkfront(self, frame, image_points):
        rotation_angle = self.detectHeadpose(frame, image_points)[1]
        if (rotation_angle <= self.mid/2 and rotation_angle >= -self.mid/2):
            return None, "REGISTER_DONE_FRONT"
        else:
            return None, None
        return None, None

    def checkright(self, frame, image_points):
        rotation_angle = self.detectHeadpose(frame, image_points)[1]
        if (rotation_angle > self.right):
            return None, "REGISTER_DONE_RIGHT"
        else:
            return None, None
        return None, None

    def detectHeadpose(self, frame, image_points):
        size = frame.shape

        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            # Left eye left corner
            (-225.0, 170.0, -135.0),
            # Right eye right corne
            (225.0, 170.0, -135.0),
            # Left Mouth corner
            (-150.0, -150.0, -125.0),
            # Right mouth corner
            (150.0, -150.0, -125.0)

        ])

        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points,
                                                                      image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        angles = 180 * \
            self.yawpitchrolldecomposition(
                cv2.Rodrigues(rotation_vector)[0])/math.pi
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array(
            [(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(frame, p1, p2, (255, 0, 0), 2)

        return angles

    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    def yawpitchrolldecomposition(self, R):

        assert (self.isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])
