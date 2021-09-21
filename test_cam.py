import cv2
import numpy as np

from OpenVinoModel import OpenVinoModel
from SCRFD.scrfd import SCRFD

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
        faces.append((face_frame, _bbox))
    return faces

video = cv2.VideoCapture(0)
scrfd = OpenVinoModel("./models/320x320_25.xml", input_size=(320, 320))
scrfd_processor = SCRFD((320, 320), 0.3)
classify = OpenVinoModel("/Users/ntdat/Downloads/20210917_classify_112_20.xml", input_size=(112,112))
while(video.isOpened()):
    _, frame = video.read()
    faces = run_face_mn(frame)
    for face, bbox in faces:
        # print(bbox)
        output = classify.predict(face)
        print(output)
        output_classify = (
        np.argmax(output[0]), np.argmax(output[1]), 1 if np.argmax(output[0]) == 0 and np.argmax(output[1]) == 0 else 0)
        color = [0,0,0]
        for i in range(3):
            color[i] = 255 if output_classify[i] == 1 else 0
        color = tuple(color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color)
        cv2.imshow("aloalo",frame)
        cv2.waitKey(1)


