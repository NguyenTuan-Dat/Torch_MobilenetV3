import os

import cv2
import numpy as np

from OpenVinoModel import OpenVinoModel
from SCRFD.scrfd import SCRFD
from modules.TDDFA.TDDFA import TDDFA_Blob
from skimage import transform as trans

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

def run_3ddfa_facial_landmarks(face_frame, bbox):
    global tddfa, landmarks
    bbox = bbox[:4]
    face_frame = (face_frame - 127.5) / 128.
    landmark_input = cv2.resize(face_frame, (120, 120))
    raw_points = landmarks.predict(landmark_input)
    param = np.array(raw_points)
    param = tddfa.postprocess(param[0][0])

    ver = tddfa.recon_vers(param=param, roi_box=bbox)
    n_points = ver.shape[1]
    list_points = []
    for i in range(n_points):
        list_points.append((int(round(ver[0, i])), int(round(ver[1, i]))))
    return (True, list_points, param)

def crop_transform(rimg, landmark, image_size=(112, 112)):
    """ warpAffine face img by landmark
    """
    assert len(landmark) == 68 or len(landmark) == 5
    assert len(landmark[0]) == 2
    if len(landmark) == 68:
        landmark5 = np.zeros((5, 2), dtype=np.float32)
        landmark5[0] = ((landmark[36][0] + landmark[39][0]) / 2, (landmark[36][1] + landmark[39][1]) / 2)
        landmark5[1] = ((landmark[42][0] + landmark[45][0]) / 2, (landmark[42][1] + landmark[45][1]) / 2)
        landmark5[2] = landmark[30]
        landmark5[3] = landmark[48]
        landmark5[4] = landmark[54]
    else:
        landmark5 = landmark
    tform = trans.SimilarityTransform()
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]],
        dtype=np.float32)
    src *= image_size[0] / 112
    src[:, 0] += 8.0
    tform.estimate(landmark5, src)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(rimg, M, (image_size[1], image_size[0]), borderValue=0.0)
    return img

video = cv2.VideoCapture(0)
scrfd = OpenVinoModel("./models/320x320_25.xml", input_size=(320, 320))
scrfd_processor = SCRFD((320, 320), 0.3)
classify = OpenVinoModel("/Users/ntdat/Downloads/20210922_Singletask_classify_112.xml", input_size=(112,112))
landmarks = OpenVinoModel("models/mb1_120x120.xml", input_size=(120, 120))
tddfa = TDDFA_Blob()
while(video.isOpened()):
    _, frame = video.read()
    faces = run_face_mn(frame)
    for face, bbox in faces:
        w,h,c = face.shape
        (lm_status, lm_points, lm_param) = run_3ddfa_facial_landmarks(face, (0,0,h,w))
        face = crop_transform(rimg=face, landmark=lm_points)
        cv2.imshow("aloalo_bbox",face)
        output = classify.predict(face)
        os.system("clear")
        print(np.round(output, 2), "====================")
        output_classify = (
        np.argmax(output[0]), np.argmax(output[1]), 1 if np.argmax(output[0]) == 0 and np.argmax(output[1]) == 0 else 0)
        color = [0,0,0]
        for i in range(3):
            color[i] = 255 if output_classify[i] == 1 else 0
        color = tuple(color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color)
        cv2.imshow("aloalo",frame)
        cv2.waitKey(1)


