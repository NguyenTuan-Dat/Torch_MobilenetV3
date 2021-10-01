import os

import cv2
import numpy as np

from OpenVinoModel import OpenVinoModel
from SCRFD.scrfd import SCRFD
from modules.TDDFA.TDDFA import TDDFA_Blob
from skimage import transform as trans
from modules.GlassesCropper import GlassesCroper
from modules.MaskCropper import MaskCroper
from modules.LandmarksCropper import LandmarksCroper


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


glasses_cropper = GlassesCroper()
mask_cropper = MaskCroper()
landmarks_cropper = LandmarksCroper()


def crop_eyes(img, landmarks, pose):
    w, h, c = img.shape
    eye_mouth_filter = np.zeros((w, h), np.uint8)

    # cal_glasses_points_status, glasses_points = glasses_cropper.calculate_glassses_points(landmarks=landmarks,
    #                                                                                       pose=pose)
    # cal_mask_points_status, mask_points = mask_cropper.calculate_mask_points(landmarks=landmarks, pose=pose)
    #
    # if cal_glasses_points_status and cal_mask_points_status:
    #     cv2.fillPoly(eye_mouth_filter, np.int32(np.array([np.array(glasses_points)])), color=255)
    #     cv2.fillPoly(eye_mouth_filter, np.int32(np.array([np.array(mask_points)])), color=255)

    cal_landmarks_points_status, landmarks_points = landmarks_cropper.calculate_landmarks_points(landmarks=landmarks,
                                                                                                 pose=pose)
    if cal_landmarks_points_status:
        cv2.fillPoly(eye_mouth_filter, np.int32(np.array([np.array(landmarks_points)])), color=255)
        dst = cv2.bitwise_and(img, img, mask=eye_mouth_filter)
        return dst
    return None


video = cv2.VideoCapture(0)
scrfd = OpenVinoModel("./models/320x320_25.xml", input_size=(320, 320))
scrfd_processor = SCRFD((320, 320), 0.5)
classify = OpenVinoModel(
    "/Users/ntdat/Documents/FaceRecognitionResearch/CompanyProject/Classify Results/Classify_Model/(Drop_Conv)20210928_MobilenetV3_0.25_Adam_112_99/112_Classify_Adam_Epoch_99_Batch_2673_91.867_98.031_Time_1632768969.3380733_checkpoint.xml",
    input_size=(112, 112))
landmarks = OpenVinoModel("models/mb1_120x120.xml", input_size=(120, 120))
tddfa = TDDFA_Blob()
count = 1000

while (video.isOpened()):
    _, frame = video.read()
    faces = run_face_mn(frame)
    for face, bbox in faces:
        w, h, c = face.shape
        (lm_status, lm_points, lm_param) = run_3ddfa_facial_landmarks(face, (0, 0, h, w))
        yaw_angle, pitch_angle, rotate_angle = tddfa.cal_pose(lm_param)
        face = crop_eyes(face, lm_points, pose=(yaw_angle, pitch_angle, rotate_angle))
        if face is None:
            continue
        _face = crop_transform(rimg=face, landmark=lm_points)
        output = classify.predict(_face)
        os.system("clear")
        print(np.round(output, 2), "====================")
        print(output[0][0], output[1][0], "====================")
        print(np.argmax(output[0][0]), np.argmax(output[1][0]), "====================")
        cv2.imshow("new_img", face)
        cv2.waitKey(1)
