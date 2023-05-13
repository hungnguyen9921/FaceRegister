import logging
import os
import io
import azure.functions as func
import json
from PIL import Image
import cv2
import numpy as np
import dlib
from math import ceil
from imutils import face_utils
from .face_detection import FaceDetection, LightFaceDetection, FaceDetectionLightRfb, LandmarkDetection
from .capregister import CaptureRegisterFace

scriptpath = os.path.abspath(__file__)
scriptdir  = os.path.dirname(scriptpath)

landmark_model = os.path.join(scriptdir,'landmarks.dat')
USER_BRIGHT_INCREMENT = 0
USER_FACE_DETECTION_THRESHOLD = 0.8
FACEMASK_DETECTION_THRESHOLD = 75
LANDMARK_MODEL = os.path.join(scriptdir,'landmarks.dat')
REC_THRESHOLD = 0.6
PROTO_RFB320 = os.path.join(scriptdir,'RFB-320.prototxt')
MODEL_RFB320 = os.path.join(scriptdir,'RFB-320.caffemodel')

image_width = 224
image_height = 224
faceDetect = LightFaceDetection(PROTO_RFB320, MODEL_RFB320, REC_THRESHOLD)
landmarkDetect = LandmarkDetection(
    FACEMASK_DETECTION_THRESHOLD, LANDMARK_MODEL)
update = CaptureRegisterFace(-18, 15, 44)
i = 0
def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_body()
        image = Image.open(io.BytesIO(req_body))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        resized_frame = cv2.resize(frame,(image_width,image_height))
        rects = faceDetect.detectFaces(resized_frame, USER_BRIGHT_INCREMENT, USER_FACE_DETECTION_THRESHOLD)
        if (len(rects) == 1):
            img_points = landmarkDetect.detectLandmarkForRegister(resized_frame, rects)
            if(i == 0):
                store_registered_imgs, status = update.checkleft(
                    resized_frame, img_points)
                print(status)
            elif(i == 1):
                store_registered_imgs, status = update.checkfront(
                    resized_frame, img_points)
                print(status)
            elif(i == 2):
                store_registered_imgs, status = update.checkright(
                    resized_frame, img_points)
                print(status)
    except ValueError:
        pass
    
    if status :
        return func.HttpResponse(json.dumps({
            'success': True
        }))
    else:
        return func.HttpResponse(json.dumps({
            'success': False
        }))
