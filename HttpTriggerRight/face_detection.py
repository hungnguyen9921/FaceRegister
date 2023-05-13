import cv2
import sys
import numpy as np
import dlib
from math import ceil
from imutils import face_utils


CAFFEMODEL = ".face_detector/res10_300x300_ssd_iter_140000.caffemodel"
PROTOTEXTPATH = ".face_detector/deploy.prototxt.txt"
LANDMARK_DETECTION_MODEL = "./landmarks.dat"
MOUTH_CASCADE_FILE = './haarcascade_mouth.xml'


class LandmarkDetection:
    def __init__(self, facemask_saturation=100, model=LANDMARK_DETECTION_MODEL, mouth_cascade_file=MOUTH_CASCADE_FILE):
        self.predictor = dlib.shape_predictor(model)
        self.facemask_saturation = facemask_saturation
        self.mouth_cascade = cv2.CascadeClassifier(mouth_cascade_file)
        # self.nose_cascade = cv2.CascadeClassifier("./device_app/submodules/face_detection/models/haarcascade_nose.xml")

    def detectLandmarkForRegister(self, frame, rects):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (len(rects) == 0):
            return None
        dlibRect = dlib.rectangle(
            rects[0][0], rects[0][1], rects[0][2], rects[0][3])
        shape = self.predictor(gray, dlibRect)

        shape = face_utils.shape_to_np(shape)
        image_points = np.array([
                                (shape[30][0], shape[30][1]),     # Nose tip
                                (shape[8][0], shape[8][1]),     # Chin
                                # Left eye left corner
                                (shape[36][0], shape[36][1]),
                                # Right eye right corne
                                (shape[45][0], shape[45][1]),
                                # Left Mouth corner
                                (shape[48][0], shape[48][1]),
                                # Right mouth corner
                                (shape[54][0], shape[54][1])
                                ], dtype="double")
        return image_points

    def faceMaskDetected(self, face):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        h, w = face.shape
        face = face[int(h*0.25):h, 0:w]
        alpha = 1.4
        beta = 5
        # cv2.imwrite('./test/mask.png', face)

        face = cv2.convertScaleAbs(face, alpha=alpha, beta=beta)
        mouth_rects = self.mouth_cascade.detectMultiScale(face, minNeighbors=5)
        # nose_rects = self.nose_cascade.detectMultiScale(face)
        # cv2.imwrite('./test/' + str(len(mouth_rects)) + '_' + str(len(nose_rects)) + '.png', face)

        if (len(mouth_rects) == 0):
            return True
        return False


class FaceDetection:
    def __init__(self, model=CAFFEMODEL, proto=PROTOTEXTPATH):
        self.rects = []
        self.net = cv2.dnn.readNetFromCaffe(proto, model)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.timer = Timer()

    def detectFaces(self, frame):
        (h, w) = frame.shape[:2]
        # blobImage convert RGB (104.0, 177.0, 123.0)
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        # passing blob through the network to detect and pridiction
        self.net.setInput(blob)
        detections = self.net.forward()
        # loop over the detections
        self.rects = []
        for i in range(0, detections.shape[2]):
            # extract the confidence and prediction
            confidence = detections[0, 0, i, 2]
            # filter detections by confidence greater than the minimum confidence
            if confidence < 0.7:
                continue
            # Determine the (x, y)-coordinates of the bounding box for the

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            self.rects.append((startX, startY, endX, endY))
            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 0, 255), 2)
        return self.rects


class LightFaceDetection:
    def __init__(self, proto, model, threshold, width=320, height=240):
        self.rects = []
        # self.net = cv2.dnn.readNetFromONNX(onnx)
        self.net = cv2.dnn.readNetFromCaffe(proto, model)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.width = width
        self.height = height
        self.image_std = 128.0
        self.center_variance = 0.1
        self.size_variance = 0.2
        self.threshold = threshold
        self.strides = [8.0, 16.0, 32.0, 64.0]
        self.min_boxes = [[10.0, 16.0, 24.0], [32.0, 48.0],
                          [64.0, 96.0], [128.0, 192.0, 256.0]]
        self.priors = self.define_img_size((self.width, self.height))

    def hard_nms(self, box_scores, iou_threshold, top_k=-1, candidate_size=200):
        scores = box_scores[:, -1]
        boxes = box_scores[:, :-1]
        picked = []
        indexes = np.argsort(scores)
        indexes = indexes[-candidate_size:]
        while len(indexes) > 0:
            current = indexes[-1]
            picked.append(current)
            if 0 < top_k == len(picked) or len(indexes) == 1:
                break
            current_box = boxes[current, :]
            indexes = indexes[:-1]
            rest_boxes = boxes[indexes, :]
            iou = self.iou_of(
                rest_boxes,
                np.expand_dims(current_box, axis=0),
            )
            indexes = indexes[iou <= iou_threshold]
        return box_scores[picked, :]

    def area_of(self, left_top, right_bottom):
        hw = np.clip(right_bottom - left_top, 0.0, None)
        return hw[..., 0] * hw[..., 1]

    def iou_of(self, boxes0, boxes1, eps=1e-5):
        overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
        overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

        overlap_area = self.area_of(overlap_left_top, overlap_right_bottom)
        area0 = self.area_of(boxes0[..., :2], boxes0[..., 2:])
        area1 = self.area_of(boxes1[..., :2], boxes1[..., 2:])
        return overlap_area / (area0 + area1 - overlap_area + eps)

    def generate_priors(self, feature_map_list, shrinkage_list, image_size, min_boxes):
        priors = []
        for index in range(0, len(feature_map_list[0])):
            scale_w = image_size[0] / shrinkage_list[0][index]
            scale_h = image_size[1] / shrinkage_list[1][index]
            for j in range(0, feature_map_list[1][index]):
                for i in range(0, feature_map_list[0][index]):
                    x_center = (i + 0.5) / scale_w
                    y_center = (j + 0.5) / scale_h

                    for min_box in min_boxes[index]:
                        w = min_box / image_size[0]
                        h = min_box / image_size[1]
                        priors.append([
                            x_center,
                            y_center,
                            w,
                            h
                        ])
        # print("priors nums:{}".format(len(priors)))
        return np.clip(priors, 0.0, 1.0)

    def center_form_to_corner_form(self, locations):
        return np.concatenate([locations[..., :2] - locations[..., 2:] / 2, locations[..., :2] + locations[..., 2:] / 2], len(locations.shape) - 1)

    def define_img_size(self, image_size):
        shrinkage_list = []
        feature_map_w_h_list = []
        for size in image_size:
            feature_map = [int(ceil(size / stride)) for stride in self.strides]
            feature_map_w_h_list.append(feature_map)

        for i in range(0, len(image_size)):
            shrinkage_list.append(self.strides)
        priors = self.generate_priors(
            feature_map_w_h_list, shrinkage_list, image_size, self.min_boxes)
        return priors

    def convert_locations_to_boxes(self, locations, priors, center_variance, size_variance):
        if len(priors.shape) + 1 == len(locations.shape):
            priors = np.expand_dims(priors, 0)
        return np.concatenate([
            locations[..., :2] * center_variance *
            priors[..., 2:] + priors[..., :2],
            np.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
        ], axis=len(locations.shape) - 1)

    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate(
                [subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = self.hard_nms(box_probs,
                                      iou_threshold=iou_threshold,
                                      top_k=top_k,
                                      )
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

    def detectFaces(self, frame, bright=60, newThreshold=None):
        rect = cv2.convertScaleAbs(frame, beta=bright)
        rect = cv2.resize(rect, (self.width, self.height))
        # rect = cv2.convertScaleAbs(rect, beta=bright)
        rect = cv2.cvtColor(rect, cv2.COLOR_BGR2RGB)
        self.net.setInput(cv2.dnn.blobFromImage(
            rect, 1 / self.image_std, (self.width, self.height), 127))
        boxes, scores = self.net.forward(["boxes", "scores"])
        boxes = np.expand_dims(np.reshape(boxes, (-1, 4)), axis=0)
        scores = np.expand_dims(np.reshape(scores, (-1, 2)), axis=0)
        boxes = self.convert_locations_to_boxes(
            boxes, self.priors, self.center_variance, self.size_variance)
        boxes = self.center_form_to_corner_form(boxes)
        if (newThreshold is None):
            newThreshold = self.threshold
        boxes, labels, probs = self.predict(
            frame.shape[1], frame.shape[0], scores, boxes, newThreshold)
        self.rects = []
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            self.rects.append((box[0], box[1], box[2], box[3]))
            cv2.rectangle(frame, (box[0], box[1]),
                          (box[2], box[3]), (0, 255, 0), 2)
        return self.rects


class FaceDetectionLightRfb:
    def __init__(self):
        self.label_path = "./submodules/face_detection/models/voc-model-labels.txt"
        self.class_names = [name.strip()
                            for name in open(self.label_path).readlines()]
        self.num_classes = len(self.class_names)
        self.candidate_size = 1000
        self.threshold = 0.7
        self.model_path = "./submodules/face_detection/models/version-RFB-320.pth"
        self.net = create_Mb_Tiny_RFB_fd(
            len(self.class_names), is_test=True, device="cuda:0")
        self.predictor = create_Mb_Tiny_RFB_fd_predictor(
            self.net, candidate_size=self.candidate_size, device="cuda:0")
        self.net.load(self.model_path)
        self.rects = []

    def detectFaces(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.predictor.predict(
            image, self.candidate_size / 2, self.threshold)
        self.rects = []
        for i in range(boxes.size(0)):
            box = boxes[i, :]
            label = f" {probs[i]:.2f}"
            self.rects.append((box[0], box[1], box[2], box[3]))
            cv2.rectangle(frame, (box[0], box[1]),
                          (box[2], box[3]), (0, 255, 0), 4)
        return self.rects
