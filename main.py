from biometric_iris import utils as iris
from env import PATH_DB
from preprocess import segmentation
from roboflow import inference

import cv2
import pandas as pd

def main():
    print("Hello from hybrid-iris-recognition!")

    df = pd.read_csv("db/CASIA-Iris-Thousand.csv")

    selected_path = df['NewPath'].loc[df.index[11070]]

    selected_path = f'{PATH_DB}{selected_path}'

    # inference.get_coords(selected_path)
    inference.get_coords_curl(selected_path)

    # img  = cv2.imread(selected_path)

    # segmentation.mask(img)

    # iris.show_details(img)

if __name__ == "__main__":
    main()
