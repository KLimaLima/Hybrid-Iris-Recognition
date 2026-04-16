from biometric_iris import utils as iris
from env import PATH_DB
from preprocess import segmentation
from roboflow import inference
from fractal import box_counting

import cv2
import pandas as pd
import numpy as np
import json
import time

def main():
    print("Hello from hybrid-iris-recognition!")

    df = pd.read_csv("db/CASIA-Iris-Thousand.csv")

    show_iris_code(df)

def get_coords(dataframe):

    dataframe = pd.read_csv("db/CASIA-Iris-Thousand.csv")

    for path_to_image in dataframe['NewPath']:

        path_to_process = f'{PATH_DB}{path_to_image}'
        print(f'Getting coords for {path_to_process}')

        inference.get_coords_curl(path_to_process)

def show_iris_code(dataframe):
    dataframe = pd.read_csv("db/CASIA-Iris-Thousand.csv")

    selected_path = dataframe['NewPath'].loc[dataframe.index[11070]]

    selected_path = f'{PATH_DB}{selected_path}'
    split_xtension = selected_path.split('.')
    json_path = f'{split_xtension[0]}.json'

    img  = cv2.imread(selected_path)

    with open(json_path, "r") as j:
        data_json = json.load(j)

    # print(data_json['predictions'][0]['class'])

    iris.show_details(img,
                    data_json['predictions'][0]['x'],
                    data_json['predictions'][0]['y'],
                    data_json['predictions'][0]['width'] / 2,
                    data_json['predictions'][1]['x'],
                    data_json['predictions'][1]['y'],
                    data_json['predictions'][1]['width'] / 2)

    points_iris = iris.unravel_iris(img,
                    data_json['predictions'][0]['x'],
                    data_json['predictions'][0]['y'],
                    data_json['predictions'][0]['width'] / 2,
                    data_json['predictions'][1]['x'],
                    data_json['predictions'][1]['y'],
                    data_json['predictions'][1]['width'] / 2)

    # Define scales
    scales = np.logspace(0, 10, num=50)

    # Compute fractal dimension
    result = box_counting(points_iris, scales, method="original")

    print("Fractal Dimension:", result["fd"])
    print("R²:", result["r_squared"])

if __name__ == "__main__":

    main()
