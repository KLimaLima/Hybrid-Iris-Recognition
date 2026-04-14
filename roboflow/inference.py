# import the inference-sdk
from inference_sdk import InferenceHTTPClient

from env import API_KEY

import json
import base64
import os
import requests

# base64 < YOUR_IMAGE.jpg | curl -d @- "https://serverless.roboflow.com/iris-09eix-rvugm/2?api_key={API_KEY}"

def get_coords_curl(path_img):

    path_split = path_img.split('.')
    filepath_no_xtension = path_split[0]
    xtension_type = path_split[1]

    # Encode the image to a base64 string.
    with open(path_img, "rb") as img:
        img_base64 = base64.b64encode(img.read())

    headers = {"Content-Type": f"image/{xtension_type}; charset=utf-8"}

    response = requests.post(f'https://serverless.roboflow.com/iris-09eix-rvugm/2?api_key={API_KEY}', headers=headers, data= img_base64)

    with open(f'{filepath_no_xtension}.json', "w") as f:

        json.dump(response.json(), f)