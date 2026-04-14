from biometric_iris import utils as iris
from env import PATH_DB
from preprocess import segmentation
from roboflow import inference

import cv2
import pandas as pd
import json

def main():
    print("Hello from hybrid-iris-recognition!")

    df = pd.read_csv("db/CASIA-Iris-Thousand.csv")

    selected_path = df['NewPath'].loc[df.index[11070]]

    selected_path = f'{PATH_DB}{selected_path}'
    split_xtension = selected_path.split('.')
    json_path = f'{split_xtension[0]}.json'

    # inference.get_coords_curl(selected_path)

    img  = cv2.imread(selected_path)

    # segmentation.mask(img)

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

    # iris.show_details(img)

if __name__ == "__main__":
    main()
