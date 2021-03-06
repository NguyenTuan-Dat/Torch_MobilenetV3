import time

from OpenVinoModel import OpenVinoModel
import os
import cv2
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

face_types_dict = dict()
face_types_dict["glasses"] = [10, 13, 16, 1, 4, 7]
face_types_dict["mask"] = [12, 15, 18, 3, 6, 9]
face_types_dict["normal"] = [22, 23, 24, 19, 20, 21]
face_types_dict["hat"] = [11, 14, 17, 2, 5, 8]

DIR = "/Users/ntdat/Downloads/DN_test_cropped_1/"
SAVE = "results/20210915_48_12"

predict_cases = np.zeros([3,3])

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
    output_classify = (np.argmax(output[0]),np.argmax(output[1]), 1 if np.argmax(output[0]) == 0 and np.argmax(output[1]) == 0 else 0)
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
    output_classify = np.zeros(3)
    output_classify[np.argmax(output)] = 1
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

INPUT_SIZE = (48,48)

classify = OpenVinoModel("models/20210915_classify_48_12.xml", input_size=INPUT_SIZE)

_class = ["glasses", "mask", "normal"]
_count = dict()
_count["glasses"] = 0
_count["mask"] = 0
_count["normal"] = 0

for subdir, dirs, files in os.walk(DIR):
    for filename in files:
        if ".jpg" not in filename:
            continue
        if filename2testcase(filename) not in face_types_dict["glasses"]:
            continue
        img = cv2.imread(os.path.join(subdir, filename))
        output = np.array(classify.predict(img))
        output = (np.argmax(output[0]), np.argmax(output[1]), 1 if np.argmax(output[0]) == 0 and np.argmax(output[1]) == 0 else 0)
        output_dir = os.path.join(SAVE, _class[np.argmax(output)])
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        _count[_class[np.argmax(output)]] += 1
        cv2.imwrite(os.path.join(output_dir, filename), img)
print(_count)

print("AVG time:", np.array(times).mean())
df_cm = pd.DataFrame(predict_cases, columns=["Glass\npredicted", "Mask\npredicted", "Normal\npredicted"],
                     index=["Glass", "Mask", "Normal"])
fig = plt.figure(figsize=(3, 3))
sn.heatmap(df_cm, annot=True, fmt=".5g")
fig.tight_layout()
plt.show()