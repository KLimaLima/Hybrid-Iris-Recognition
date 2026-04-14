# import the inference-sdk
from inference_sdk import InferenceHTTPClient

from env import API_KEY

import json
import base64
import os
import requests

def get_coords(path_img):

    path_split = path_img.split('.')

    filename_no_xtension = path_split[0]
    # path_dir = f"coords/{path_split[-3]}/{path_split[-2]}"
    # path_full = f"{path_dir}/{path_split[-1]}.json"

    # if os.path.exists(path_full):

    #     print(f"skipping {path_full}")
    #     return True

    # # checking if the directory exist or not.
    # if not os.path.exists(path_dir):
        
    #     # if the directory is not present then create it.
    #     os.makedirs(path_dir)

    print(f'getting roi for: {path_img}')

    # # initialize the client
    # CLIENT = InferenceHTTPClient(
    #     api_url="https://serverless.roboflow.com",
    #     api_key="CzEmp69liCYJnY8ycp95"
    # )

    # # infer on a local image
    # result = CLIENT.infer(path_img, model_id="iris-09eix-rvugm/2")

    # 1. Import the library
    from inference_sdk import InferenceHTTPClient

    # 2. Connect to your workflow
    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="KRpj6UJRB7tdh8rzAalz"
    )

    # 3. Run your workflow on an image
    result = client.run_workflow(
        workspace_name="camouflaged-animals-ai-object-detection-workshop",
        workflow_id="detect-and-classify",
        images={
            "image": 'image.png' # Path to your image file
        }
        # use_cache=True # Speeds up repeated requests
    )

    # 4. Get your results
    print(result)


    with open(f'{path_split}.json', "w") as f:

        json.dump(result, f)

    return False

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
    # print(response.content)

    with open(f'{filepath_no_xtension}.json', "w") as f:

        json.dump(response.json(), f)