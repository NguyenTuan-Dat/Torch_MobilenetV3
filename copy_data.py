import argparse
import os
import shutil
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--cropped_faces_folder", type= str)
parser.add_argument("-s", "--path_to_save", type=str, default=None)
parser.add_argument("-f", "--face_types", default=None, nargs='+')
parser.add_argument("-hp", "--headposes", default=None, nargs='+')
args = parser.parse_args()


PATH_TO_SAVE = args.path_to_save
FROM = args.cropped_faces_folder

face_types_dict = dict()
face_types_dict["glasses"] = [10, 13, 16, 1, 4, 7]
face_types_dict["mask"] = [12, 15, 18, 3, 6, 9]
face_types_dict["normal"] = [22, 23, 24, 19, 20, 21]
face_types_dict["hat"] = [11, 14, 17, 2, 5, 8]

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

def find_using_testcase(face_types=("normal", "glasses", "mask"), using_headpose=(0, 1, 2)):
    global face_types_dict

    headpose_dict = [
        [22, 19, 12, 3, 10, 1, 11, 2],  # rotate angle: abs <= 10
        [23, 20, 15, 6, 13, 4, 14, 5],  # rotate angle: 10 < abs <= 30
        [24, 21, 18, 9, 16, 7, 17, 8],  # rotate angle: 30 < abs <=45
    ]

    face_type_testcases = []
    for face_type in face_types:
        face_type_testcases.extend(face_types_dict[face_type])

    headpose_testcase = []
    for headpose in using_headpose:
        headpose_testcase.extend(headpose_dict[headpose])

    using_testcase = np.intersect1d(face_type_testcases, headpose_testcase)
    print(using_testcase)
    return using_testcase

for face_type in args.face_types:
    for headpose in args.headposes:
        paths = []
        labels = []
        features = []
        eq_scores = []

        print(face_type)
        using_testcase = find_using_testcase(face_types=(face_type,), using_headpose=(int(headpose),))

        path_to_save = os.path.join(PATH_TO_SAVE, face_type)
        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)

        for idx, dir in enumerate(os.listdir(FROM)):
            if dir == ".DS_Store":
                continue
            path_to_dir = os.path.join(FROM, dir)
            for img_name in os.listdir(path_to_dir):
                if img_name == ".DS_Store":
                    continue
                if filename2testcase(img_name) not in using_testcase:
                    # print(img_name, "no run")
                    continue
                shutil.copy(os.path.join(path_to_dir, img_name), os.path.join(path_to_save, img_name))