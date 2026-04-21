from biometric_iris import utils as iris
from env import PATH_DB
from preprocess import segmentation
from roboflow import inference
import fractal

import cv2
import pandas as pd
import numpy as np
import json
import time

def main():
    print("Hello from hybrid-iris-recognition!")

    df = pd.read_csv("db/CASIA-Iris-Thousand.csv")

    selected_path = df['NewPath'].loc[df.index[11070]] # 11070

    selected_path = f'{PATH_DB}{selected_path}'

    # show_iris_code(df)

    # get_data_np(selected_path)

    npz_path = f'{selected_path.split('.')[0]}.npz'
    npz_file = np.load(npz_path, allow_pickle=True)
    my_dict = npz_file['fractal_dimension'].item()
    print(my_dict['box_counting_original'])

def get_data_np(img_path: str):

    split_xtenstion = img_path.split('.')
    path_json = f'{split_xtenstion[0]}.json'

    img_ori = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    with open(path_json, "r") as j:
        data_json = json.load(j)

    code_iris = iris.unravel_iris(img_ori,
                    data_json['predictions'][0]['x'],
                    data_json['predictions'][0]['y'],
                    data_json['predictions'][0]['width'] / 2,
                    data_json['predictions'][1]['x'],
                    data_json['predictions'][1]['y'],
                    data_json['predictions'][1]['width'] / 2)
    
    dict_fractal = calc_fractal(code_iris)

    # np_fractal = dict2numpy(dict_fractal)

    np.savez_compressed(f'{split_xtenstion[0]}.npz',
                        img_original= img_ori,
                        code_iris= code_iris,
                        fractal_dimension= dict_fractal)

def get_coords(dataframe):

    dataframe = pd.read_csv("db/CASIA-Iris-Thousand.csv")

    for path_to_image in dataframe['NewPath']:

        path_to_process = f'{PATH_DB}{path_to_image}'
        print(f'Getting coords for {path_to_process}')

        inference.get_coords_curl(path_to_process)

def show_iris_code(dataframe):
    dataframe = pd.read_csv("db/CASIA-Iris-Thousand.csv")

    selected_path = dataframe['NewPath'].loc[dataframe.index[11070]] # 11070

    selected_path = f'{PATH_DB}{selected_path}'
    split_xtension = selected_path.split('.')
    json_path = f'{split_xtension[0]}.json'

    img  = cv2.imread(selected_path, cv2.IMREAD_GRAYSCALE)

    print(img.shape)

    with open(json_path, "r") as j:
        data_json = json.load(j)

    # print(data_json['predictions'][0]['class'])

    # NOTE: to use show_details, must cv2.imread(img_path) - cannot use grayscale
    # iris.show_details(img,
    #                 data_json['predictions'][0]['x'],
    #                 data_json['predictions'][0]['y'],
    #                 data_json['predictions'][0]['width'] / 2,
    #                 data_json['predictions'][1]['x'],
    #                 data_json['predictions'][1]['y'],
    #                 data_json['predictions'][1]['width'] / 2)

    points_iris = iris.unravel_iris(img,
                    data_json['predictions'][0]['x'],
                    data_json['predictions'][0]['y'],
                    data_json['predictions'][0]['width'] / 2,
                    data_json['predictions'][1]['x'],
                    data_json['predictions'][1]['y'],
                    data_json['predictions'][1]['width'] / 2)
    
    print(points_iris.shape)

    calc_fractal(points_iris, True)

    # # Define scales
    # scales = np.logspace(0, 10, num=50)

    # # Compute fractal dimension
    # result = fractal.box_counting(points_iris, scales, method="original")

    # print("Fractal Dimension:", result["fd"])
    # print("R²:", result["r_squared"])

def calc_fractal(code_iris, show= False):

    scales=np.logspace(0, 10, num=50)
    # oversample_rate=10
    res={}
    for method in ["original"]:
        r=fractal.box_counting(code_iris,scales,method=method, return_boxes=True)
        res[f"box_counting_{method}"]=r

    r=fractal.temporal_sampling(code_iris, min_step=1, max_step=2, return_boxes=True)
    res["temporal_sampling"]=r    

    scales=0.007*np.exp(np.linspace(np.log(1),np.log(3),5))
    r=fractal.corr_sum(code_iris, scales, return_boxes=True)
    res["corr_sum"]=r

    r=fractal.corr_sum_takens(code_iris)
    res["corr_sum_takens"]=r

    if show:
        for key,value in res.items():
            print(f'{key}: {value}')

    return res

def dict2numpy(my_data: dict):

    res = my_data.items()

    data = list(res)

    np_array = np.array(data)

    return np_array

if __name__ == "__main__":

    main()
