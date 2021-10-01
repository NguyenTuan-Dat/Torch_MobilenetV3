import time

from OpenVinoModel import OpenVinoModel
import os
import cv2
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from modules.LandmarksCropper import LandmarksCroper
from modules.TDDFA.TDDFA import TDDFA_Blob

face_types_dict = dict()
face_types_dict["glasses"] = [10, 13, 16, 1, 4, 7]
face_types_dict["mask"] = [12, 15, 18, 3, 6, 9]
face_types_dict["normal"] = [22, 23, 24, 19, 20, 21]
face_types_dict["hat"] = [11, 14, 17, 2, 5, 8]

DIR = "/Users/ntdat/Downloads/DN_test_cropped_1"

predict_cases = np.zeros([3, 3])

times = []


def filename2testcase(file_name):
    """
    filename format:
        <id>_<glass>_<hat>_<mask>_<brightness>_<angle_range>_<time>
    """

    PREFIX_2_TESTCASE = {
        "_0_0_0_0_0_": 22,
        "_0_0_0_0_1_": 23,
        "_0_0_0_0_2_": 24,
        "_0_0_0_brightness_0_": 19,
        "_0_0_0_brightness_1_": 20,
        "_0_0_0_brightness_2_": 21,
        "_0_0_mask_0_0_": 12,
        "_0_0_mask_0_1_": 15,
        "_0_0_mask_0_2_": 18,
        "_0_0_mask_brightness_0_": 3,
        "_0_0_mask_brightness_1_": 6,
        "_0_0_mask_brightness_2_": 9,
        "_0_glass_0_0_0_": 10,
        "_0_glass_0_0_1_": 13,
        "_0_glass_0_0_2_": 16,
        "_0_glass_0_brightness_0_": 1,
        "_0_glass_0_brightness_1_": 4,
        "_0_glass_0_brightness_2_": 7,
        "_hat_0_0_0_0_": 11,
        "_hat_0_0_0_1_": 14,
        "_hat_0_0_0_2_": 17,
        "_hat_0_0_brightness_0_": 2,
        "_hat_0_0_brightness_1_": 5,
        "_hat_0_0_brightness_2_": 8
    }

    for prefix in PREFIX_2_TESTCASE.keys():
        if prefix in file_name:
            return PREFIX_2_TESTCASE[prefix]
    return None


def multitask_to_true_false_cases(filename, output):
    global face_types_dict, predict_cases
    output_classify = (
        np.argmax(output[0]), np.argmax(output[1]), 1 if np.argmax(output[0]) == 0 and np.argmax(output[1]) == 0 else 0)
    if filename2testcase(filename) in face_types_dict["glasses"]:
        if output_classify[0] == 1:
            predict_cases[0][0] += 1
        else:
            predict_cases[0][1] += 1 if output_classify[1] == 1 else 0
            predict_cases[0][2] += 1 if output_classify[1] != 1 else 0
    elif filename2testcase(filename) in face_types_dict["mask"]:
        if output_classify[1] == 1:
            predict_cases[1][1] += 1
        else:
            predict_cases[1][0] += 1 if output_classify[0] == 1 else 0
            predict_cases[1][2] += 1 if output_classify[0] != 1 else 0
    elif filename2testcase(filename) in face_types_dict["normal"]:
        if output_classify[2] == 1:
            predict_cases[2][2] += 1
        else:
            predict_cases[2][1] += 1 if output_classify[1] == 1 else 0
            predict_cases[2][0] += 1 if output_classify[1] != 1 else 0


def single_task_to_false_cases(filename, output):
    global face_types_dict, predict_cases
    output_classify = (1 if output[0] > 0.5 else 0,
                       1 if output[1] > 0.5 else 0,
                       1 if output[0] <= 0.5 and output[1] <= 0.5 else 0)
    if filename2testcase(filename) in face_types_dict["glasses"]:
        if output_classify[0] == 1:
            predict_cases[0][0] += 1
        else:
            predict_cases[0][1] += 1 if output_classify[1] == 1 else 0
            predict_cases[0][2] += 1 if output_classify[1] != 1 else 0
    elif filename2testcase(filename) in face_types_dict["mask"]:
        if output_classify[1] == 1:
            predict_cases[1][1] += 1
        else:
            predict_cases[1][0] += 1 if output_classify[0] == 1 else 0
            predict_cases[1][2] += 1 if output_classify[0] != 1 else 0
    elif filename2testcase(filename) in face_types_dict["normal"]:
        if output_classify[2] == 1:
            predict_cases[2][2] += 1
        else:
            predict_cases[2][1] += 1 if output_classify[1] == 1 else 0
            predict_cases[2][0] += 1 if output_classify[1] != 1 else 0


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


landmarks_cropper = LandmarksCroper()


def crop_eyes(img, landmarks, pose):
    w, h, c = img.shape
    eye_mouth_filter = np.zeros((w, h), np.uint8)

    cal_landmarks_points_status, landmarks_points = landmarks_cropper.calculate_landmarks_points(landmarks=landmarks,
                                                                                                 pose=pose)
    if cal_landmarks_points_status:
        cv2.fillPoly(eye_mouth_filter, np.int32(np.array([np.array(landmarks_points)])), color=255)
        dst = cv2.bitwise_and(img, img, mask=eye_mouth_filter)
        return dst
    return None


INPUT_SIZE = (112, 112)

classify = OpenVinoModel(
    "/Users/ntdat/Downloads/112_Classify_Adam_Epoch_138_Batch_3864_94.916_98.475_Time_1632990423.1972141_checkpoint.xml",
    input_size=INPUT_SIZE)
landmarks = OpenVinoModel("models/mb1_120x120.xml", input_size=(120, 120))
tddfa = TDDFA_Blob()

for subdir, dirs, files in os.walk(DIR):
    for filename in files:
        if ".jpg" not in filename:
            continue
        face = cv2.imread(os.path.join(subdir, filename))

        w, h, c = face.shape
        (lm_status, lm_points, lm_param) = run_3ddfa_facial_landmarks(face, (0, 0, h, w))
        yaw_angle, pitch_angle, rotate_angle = tddfa.cal_pose(lm_param)
        face = crop_eyes(face, lm_points, pose=(yaw_angle, pitch_angle, rotate_angle))
        if face is None:
            continue
        face = crop_transform(rimg=face, landmark=lm_points)
        t = time.time()
        output = np.array(classify.predict(face))
        times.append(time.time() - t)
        multitask_to_true_false_cases(filename, output)
        # single_task_to_false_cases(filename, output[0][0])
print("AVG time:", np.array(times).mean())
df_cm = pd.DataFrame(predict_cases, columns=["Glass\npredicted", "Mask\npredicted", "Normal\npredicted"],
                     index=["Glass", "Mask", "Normal"])
fig = plt.figure(figsize=(3, 3))
sn.heatmap(df_cm, annot=True, fmt=".5g")
fig.tight_layout()
plt.show()
