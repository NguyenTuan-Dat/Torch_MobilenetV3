import cv2
import numpy

from MobilenetV3 import mobilenetv3_small
from OpenVinoModel import OpenVinoModel
from SCRFD.scrfd import SCRFD
import torch
import torchvision.transforms as transforms

from config import config

def run_face_mn(image_frame=None):
    faces = []

    output = scrfd.predict(image_frame)
    bboxes = scrfd_processor.processing_output(output)
    bbox_counter = 0
    faces = []
    for bbox in bboxes:
        bbox_counter += 1

        _bbox = scrfd_processor.get_face_location(bbox, image_frame.shape)
        # print("_bbox", _bbox)

        custom_bbox = [_bbox[0], _bbox[1], _bbox[2] -
                       _bbox[0], _bbox[3] - _bbox[1]]

        face_frame = image_frame[_bbox[1]: _bbox[3], _bbox[0]: _bbox[2]]
        faces.append(face_frame)
    return faces

video = cv2.VideoCapture(0)
scrfd = OpenVinoModel("./models/320x320_25.xml", input_size=(320, 320))
scrfd_processor = SCRFD((320, 320), 0.2)
classify = OpenVinoModel("/Users/ntdat/Downloads/20210908_classify_4.xml", input_size=(112, 112))
while(video.isOpened()):
    _, frame = video.read()
    faces = run_face_mn(frame)
    for face in faces:
        output = classify.predict(face)
        print("asdsdads", numpy.array(output).round(2))

