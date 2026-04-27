# import the inference-sdk
from inference_sdk import InferenceHTTPClient

from env import API_KEY

import json
import base64
import os
from pathlib import Path
import requests
import time

# base64 < YOUR_IMAGE.jpg | curl -d @- "https://serverless.roboflow.com/iris-09eix-rvugm/2?api_key={API_KEY}"

def get_coords_curl(path_img):

    path_split = path_img.split('.')
    filepath_no_xtension = path_split[0]
    xtension_type = path_split[1]
    filepath_json = f'{filepath_no_xtension}.json'

    if Path(filepath_json).exists():
        print('JSON file already exist')
        return True

    # Encode the image to a base64 string.
    with open(path_img, "rb") as img:
        img_base64 = base64.b64encode(img.read())

    headers = {"Content-Type": f"image/{xtension_type}; charset=utf-8"}

    response = requests.post(f'https://serverless.roboflow.com/iris-09eix-rvugm/2?api_key={API_KEY}', headers=headers, data= img_base64)

    if not response.ok:
        print("Problem getting response from Roboflow")
        return False

    with open(filepath_json, "w") as f:

        json.dump(response.json(), f)

    time.sleep(2)

    return True