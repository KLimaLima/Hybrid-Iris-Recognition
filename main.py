from biometric_iris import utils as iris
from env import PATH_DB
from preprocess import segmentation
from roboflow import inference
from fractal import fractal, utils

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time

def main():
    print("Hello from hybrid-iris-recognition!")

    df = pd.read_csv("db/CASIA-Iris-Thousand.csv")

    # for path_to_image in df['NewPath']:

    #     path_to_process = f'{PATH_DB}{path_to_image}'
    #     print(f'Processing {path_to_process}')

    selected_path = df['NewPath'].loc[df.index[11070]] # 11070

    selected_path = f'{PATH_DB}{selected_path}'
    
    show(selected_path)

    # get_data_np(selected_path)

    # npz_path = f'{selected_path.split('.')[0]}.npz'
    # npz_file = np.load(npz_path, allow_pickle=True)
    # cv2.imshow('image from npz file', npz_file['img_original'])
    # cv2.waitKey(100)
    # my_dict = npz_file['fractal_dimension'].item()
    # print(type(my_dict['box_counting_original']['boxes'])) # numpy array

def extract_save2npz(img_path: str):

    # get coords from JSON
    split_xtenstion = img_path.split('.')
    path_json = f'{split_xtenstion[0]}.json'

    with open(path_json, "r") as j:
        data_json = json.load(j)

    pupil = {'x': data_json['predictions'][0]['x'],
                  'y': data_json['predictions'][0]['y'],
                  'r': data_json['predictions'][0]['width'] / 2}
    
    limbus = {'x': data_json['predictions'][1]['x'],
                  'y': data_json['predictions'][1]['y'],
                  'r': data_json['predictions'][1]['width'] / 2}
    
    # ORIGINAL get code and fractal
    img_ori = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    code_original = iris.unravel_iris(img_ori,
                                  limbus['x'], limbus['y'], limbus['r'],
                                  pupil['x'], pupil['y'], pupil['r'])
    
    fd_original = calc_fractal(code_original)

    # HIST EQ get code and fractal
    img_hist_eq = cv2.equalizeHist(img_ori)

    code_hist_eq = iris.unravel_iris(img_hist_eq,
                                  limbus['x'], limbus['y'], limbus['r'],
                                  pupil['x'], pupil['y'], pupil['r'])
    
    fd_hist_eq = calc_fractal(code_hist_eq)

    # save everything to npz file
    np.savez_compressed(f'{split_xtenstion[0]}.npz',
                        pupil= pupil,
                        limbus= limbus,
                        img_original= img_ori,
                        code_original= code_original,
                        fd_original= fd_original,
                        img_hist_eq= img_hist_eq,
                        code_hist_eq= code_hist_eq,
                        fd_hist_eq= fd_hist_eq)

def get_coords(dataframe):

    dataframe = pd.read_csv("db/CASIA-Iris-Thousand.csv")

    for path_to_image in dataframe['NewPath']:

        path_to_process = f'{PATH_DB}{path_to_image}'
        print(f'Getting coords for {path_to_process}')

        inference.get_coords_curl(path_to_process)

def show(path_img: str):

    split_xtension = path_img.split('.')
    json_path = f'{split_xtension[0]}.json'

    img  = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)

    with open(json_path, "r") as j:
        data_json = json.load(j)

    pupil_data = {'x': data_json['predictions'][0]['x'],
                  'y': data_json['predictions'][0]['y'],
                  'r': data_json['predictions'][0]['width'] / 2}
    
    limbus_data = {'x': data_json['predictions'][1]['x'],
                  'y': data_json['predictions'][1]['y'],
                  'r': data_json['predictions'][1]['width'] / 2}

    code_iris = iris.unravel_iris(img,
                                  limbus_data['x'], limbus_data['y'], limbus_data['r'],
                                  pupil_data['x'], pupil_data['y'], pupil_data['r'])
    
    calc_fractal(code_iris, True)

    img_hist_eq = cv2.equalizeHist(img)

    code_iris_histeq = iris.unravel_iris(img_hist_eq,
                                  limbus_data['x'], limbus_data['y'], limbus_data['r'],
                                  pupil_data['x'], pupil_data['y'], pupil_data['r'])
    
    calc_fractal(code_iris_histeq, True)

    print(limbus_data)

    cv2.circle(img, (int(limbus_data['x']), int(limbus_data['y'])), int(limbus_data['r']), (255, 0, 0), 3)
    cv2.circle(img, (int(limbus_data['x']), int(limbus_data['y'])), 2, (255, 0, 0), 2)
    cv2.circle(img, (int(pupil_data['x']), int(pupil_data['y'])), int(pupil_data['r']), (0, 255, 0), 3)
    cv2.circle(img, (int(pupil_data['x']), int(pupil_data['y'])), 2, (0, 255, 0), 2)

    cv2.circle(img_hist_eq, (int(limbus_data['x']), int(limbus_data['y'])), int(limbus_data['r']), (255, 0, 0), 3)
    cv2.circle(img_hist_eq, (int(limbus_data['x']), int(limbus_data['y'])), 2, (255, 0, 0), 2)
    cv2.circle(img_hist_eq, (int(pupil_data['x']), int(pupil_data['y'])), int(pupil_data['r']), (0, 255, 0), 3)
    cv2.circle(img_hist_eq, (int(pupil_data['x']), int(pupil_data['y'])), 2, (0, 255, 0), 2)

    f, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(img, cmap=plt.cm.gray)
    axes[0, 0].set_title('Original Iris')
    axes[0, 1].imshow(code_iris, cmap=plt.cm.gray)
    axes[0, 1].set_title('Original Code')

    axes[1, 0].imshow(img_hist_eq, cmap=plt.cm.gray)
    axes[1, 0].set_title('Hist EQ Iris')
    axes[1, 1].imshow(code_iris_histeq, cmap=plt.cm.gray)
    axes[1, 1].set_title('Hist EQ Code')

    plt.show()

def calc_fractal(code_iris, show= False):

    scales=np.logspace(0, 10, num=50)
    # oversample_rate=10
    res = {}

    r = fractal.box_counting(code_iris, scales, method='original', return_boxes=True)
    res["box_counting_original"] = r

    r = fractal.temporal_sampling(code_iris, min_step=1, max_step=2, return_boxes=True)
    res["temporal_sampling"] = r    

    scales = 0.007*np.exp(np.linspace(np.log(1),np.log(3),5))
    r = fractal.corr_sum(code_iris, scales, return_boxes=True)
    res["corr_sum"] = r

    r = fractal.corr_sum_takens(code_iris)
    res["corr_sum_takens"] = r

    if show:
        for key,value in res.items():
            print(f'{key}: {value}')

    return res

if __name__ == "__main__":

    main()
