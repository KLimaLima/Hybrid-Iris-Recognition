from biometric_iris import utils as iris
from env import PATH_DB
from preprocess import segmentation
from roboflow import inference
from fractal import fractal, utils
from my_keras import model

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import math

def main():
    print("Hello from hybrid-iris-recognition!")

    df = pd.read_csv("db/CASIA-Iris-Thousand.csv")

    # Source - https://stackoverflow.com/a/71769532
    # Posted by Devesh
    # Retrieved 2026-04-27, License - CC BY-SA 4.0

    df = df.query("Index <= 500") # >= 10500

    # Shuffle your dataset 
    shuffle_df = df.sample(frac=1)

    # Define a size for your train set 
    train_size = int(0.8 * len(df))

    # Split your dataset 
    train_set = shuffle_df[:train_size]
    test_set = shuffle_df[train_size:]

    (X_gabor1, X_gabor2, X_gabor3, X_gabor4, X_fd), y = prepare_input(train_set)

    unique, counts = np.unique(y, return_counts=True)
    # print(y.size)
    from tensorflow.keras.utils import to_categorical
    labels = to_categorical(y, num_classes=y.size)
    # print(labels)
    print(y.dtype)
    # exit()

    model.make_model((150, 300), (4,), y.size,
                     X_gabor1, X_gabor2, X_gabor3, X_gabor4, X_fd, labels)

    # temporary
    # df = df.query("Index >= 10760") # 1380

    # for path_to_image in df['NewPath']:

    #     path_to_process = f'{PATH_DB}{path_to_image}'
    #     print(f'Processing {path_to_process}')
    #     extract_save2npz(path_to_process)

    # selected_path = df['NewPath'].loc[df.index[11070]] # 11070
    # selected_path = f'{PATH_DB}{selected_path}'
    # extract_save2npz(selected_path)
    
    # npz_path = f'{selected_path.split('.')[0]}.npz'
    # npz_file = np.load(npz_path, allow_pickle=True)

    # dict_fd = npz_file['fd_hist_eq'].item()
    # dict_gabor_phase = npz_file['gabor_phase'].item()
    # dict_gabor_magnitude = npz_file['gabor_magnitude'].item()

    # print(dict_fd.keys()) # numpy float64

    # gabor_orientations = dict_gabor_phase.keys()
    # for val in dict_gabor_phase.values():
    #     print(val) # numpy array

def prepare_input(df):
    
    X_fd = []
    X_gabor1, X_gabor2, X_gabor3, X_gabor4 = [],[],[],[]
    y = []
    
    for npz_path in df['NPZ_Path']:
        
        npz_file = np.load(npz_path, allow_pickle=True)

        dict_gabor_phase = npz_file['gabor_phase'].item()
        gabor_phase_list = list(dict_gabor_phase.values())
        X_gabor1.append(gabor_phase_list[0])
        X_gabor2.append(gabor_phase_list[1])
        X_gabor3.append(gabor_phase_list[2])
        X_gabor4.append(gabor_phase_list[3])

        dict_fd = npz_file['fd_hist_eq'].item()
        temp_list = []
        temp_list.append(dict_fd['box_counting_original']['fd'])
        temp_list.append(dict_fd['temporal_sampling']['fd'])
        temp_list.append(dict_fd['corr_sum']['fd'])
        temp_list.append(dict_fd['corr_sum_takens'])
        fd_tuple = tuple(temp_list)
        X_fd.append(fd_tuple)
        
        y.append(f'{npz_file['person_num']}-{npz_file['lr']}')

    # Source - https://stackoverflow.com/a/70051602
    # Posted by j1-lee, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-04-28, License - CC BY-SA 4.0

    d = {x: i for i, x in enumerate(set(y))}
    lst_new = [d[x] for x in y]
    # print(lst_new)
    y = lst_new

    X_gabor1, X_gabor2, X_gabor3, X_gabor4 = np.array(X_gabor1), np.array(X_gabor2), np.array(X_gabor3), np.array(X_gabor4)
    X_fd = np.array(X_fd)
    # print(type(y[0]))
    y = np.array(y, dtype=np.float32)

    # print(X_gabor1.dtype)
    # print(X_gabor2.dtype)
    # print(X_gabor3.dtype)
    # print(X_gabor4.dtype)
    # print(fd_tuple)
    # print(y.dtype)
    # exit()

    return (X_gabor1, X_gabor2, X_gabor3, X_gabor4, X_fd), y

def extract_save2npz(img_path: str):

    # get iris properties/labels
    split_slash = img_path.split('/')
    lr = split_slash[-2]
    person_num = split_slash[-3]

    # get coords from JSON
    split_xtenstion = img_path.split('.')
    path_json = f'{split_xtenstion[0]}.json'

    # if there is no existing coords, get it
    try:    
        with open(path_json, "r") as j:
            data_json = json.load(j)
    except:
        print(f'No coords exist for {img_path}, getting its coords')
        inference.get_coords_curl(img_path)

        with open(path_json, "r") as j:
            data_json = json.load(j)
            
    # putting corresponding coords
    try:
        pupil = {'x': data_json['predictions'][0]['x'],
                    'y': data_json['predictions'][0]['y'],
                    'r': data_json['predictions'][0]['width'] / 2}
        
        limbus = {'x': data_json['predictions'][1]['x'],
                    'y': data_json['predictions'][1]['y'],
                    'r': data_json['predictions'][1]['width'] / 2}
    except:
        print('Not enough data from json')
        return
    
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

    # GABOR FILTER get magnitude and phase
    gabor_magnitude = {}
    gabor_phase ={}
    
    for theta in [0, math.pi/4, math.pi/2, 3*math.pi/4,]:

        my_gabor, magnitude, phase = gabor_v2(code_hist_eq, theta)

        gabor_magnitude[theta] = magnitude
        gabor_phase[theta] = phase

        # f, axes = plt.subplots(2, 2, figsize=(8, 8))
        # axes[0, 0].imshow(code_hist_eq, cmap=plt.cm.gray)
        # axes[0, 0].set_title('Code HIST EQ')
        # axes[0, 1].imshow(, cmap=plt.cm.gray)
        # axes[0, 1].set_title('Gabor HIST EQ')

        # axes[1, 0].imshow(gabor_magnitude[theta], cmap=plt.cm.gray)
        # axes[1, 0].set_title(f'Gabor Magnitude {theta}')
        # axes[1, 1].imshow(gabor_phase[theta], cmap=plt.cm.gray)
        # axes[1, 1].set_title(f'Gabor Phase {theta}')

        # fd_gb_phase = calc_fractal(gabor_phase[theta], True) # calc fd with phase gets very reliable value

        # plt.show()
        # cv2.waitKey(0)

    # save everything to npz file
    np.savez_compressed(f'{split_xtenstion[0]}.npz',
                        
                        original_path= img_path,
                        person_num= person_num,
                        lr= lr,
                        
                        pupil= pupil,
                        limbus= limbus,

                        img_original= img_ori,
                        code_original= code_original,
                        fd_original= fd_original, # dict

                        img_hist_eq= img_hist_eq,
                        code_hist_eq= code_hist_eq,
                        fd_hist_eq= fd_hist_eq, # dict
                        
                        gabor_magnitude= gabor_magnitude, # dict
                        gabor_phase= gabor_phase) # dict

#############################################################

# Source - https://stackoverflow.com/a/61332913
# Posted by Cris Luengo, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-26, License - CC BY-SA 4.0

# image = np.zeros((64, 64))
# image[32, 32] = 1          # a delta impulse image to visualize the filtering kernel

def gabor_v2(image, theta_in_rad, wavelength = 10, gamma = 0.5):

    # line below converts degree to radian
    # orientation = -theta_in_degree / 180 * math.pi    # in radian, and seems to run in opposite direction
    orientation = theta_in_rad
    sigma = 0.5 * wavelength * 1         # 1 == SpatialFrequencyBandwidth
    # gamma = 0.5                          # SpatialAspectRatio
    shape = 1 + 2 * math.ceil(4 * sigma) # smaller cutoff is possible for speed
    shape = (shape, shape)
    gabor_filter_real = cv2.getGaborKernel(shape, sigma, orientation, wavelength, gamma, psi=0)
    gabor_filter_imag = cv2.getGaborKernel(shape, sigma, orientation, wavelength, gamma, psi=math.pi/2)

    gabor = cv2.filter2D(image, -1, gabor_filter_real) + 1j * cv2.filter2D(image, -1, gabor_filter_imag)
    mag = np.abs(gabor)
    phase = np.angle(gabor)

    return gabor, mag, phase

#############################################################

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
