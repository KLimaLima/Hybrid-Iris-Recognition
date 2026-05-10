import pandas as pd
import numpy as np
import csv
import os
import cv2

from main import get_coords, gabor_v2, gaborv2_and_create, extract_save2npz

def main():

    df = pd.read_csv("db/CASIA-Iris-Thousandv2.csv")

    df = df.query("Class_num <= 100")

    # temp = np.load('input_img_gabor.npy')

    # print(temp.shape)

    prep_2(df)

    # print(df)

    # add_fd_2_csv()
    # get_coords(df)
    # create_gabor_img(df)
    # for img_path in df['Img_Path']:
    #     extract_save2npz(img_path)

    # print(os.walk('/media/limalima/D/Database02'))
    # for root, dirs, files in os.walk('/media/limalima/D/Database02'):
    #     print("Current directory:", root)
    #     print("Subdirectories:", dirs)
    #     print("Files:", files)
    #     print('----------------')

def prep_1(df):

    input_img_gabor = []
    input_fd = []
    input_iris_code = []
    output_label = []

    for row in df.itertuples(index=False):

        npz_file = np.load(row.NPZ_Path, allow_pickle=True)
        code_hist = npz_file['code_hist_eq']

        label = int(row.Class_num)
        lr = row.Class_lr
        if lr == 'L':
            label += 100
        elif lr != 'R':
            print(f'lr is wrong for {row.Img_Path}')
            exit()

        fd_tuple = (row.fd_box_counting, row.fd_temporal_sampling, row.fd_corr_sum)

        for theta in [0, 45, 90, 135]:
            gabor, mag, phase = gabor_v2(code_hist, theta)
            del gabor
            del phase

            cv2.imwrite('temp.jpg', mag)
            mag = cv2.imread('temp.jpg')

            output_label.append(label)
            input_fd.append(fd_tuple)
            input_img_gabor.append(mag)

            os.remove('temp.jpg')

    input_img_gabor = np.array(input_img_gabor)
    input_fd = np.array(input_fd)
    output_label = np.array(output_label)

    # return input_img_gabor, input_fd, label

    np.save('input_img_gabor.npy', input_img_gabor)
    np.save('input_fd.npy', input_fd)
    np.save('label.npy', output_label)

def prep_2(df):

    input_img_gabor1 = []
    input_img_gabor2 = []
    input_img_gabor3 = []
    input_img_gabor4 = []
    input_iris_code = []
    input_fd = []
    output_label = []

    for row in df.itertuples(index=False):

        npz_file = np.load(row.NPZ_Path, allow_pickle=True)
        code_hist = npz_file['code_hist_eq']

        label = int(row.Class_num)
        lr = row.Class_lr
        if lr == 'L':
            label += 100
        elif lr != 'R':
            print(f'lr is wrong for {row.Img_Path}')
            exit()

        fd_tuple = (row.fd_box_counting, row.fd_temporal_sampling, row.fd_corr_sum)

        gabor, mag, phase = gabor_v2(code_hist, 0)
        del gabor
        del phase
        cv2.imwrite('temp.jpg', mag)
        input_img_gabor1.append(cv2.imread('temp.jpg'))
        os.remove('temp.jpg')

        gabor, mag, phase = gabor_v2(code_hist, 45)
        del gabor
        del phase
        cv2.imwrite('temp.jpg', mag)
        input_img_gabor2.append(cv2.imread('temp.jpg'))
        os.remove('temp.jpg')

        gabor, mag, phase = gabor_v2(code_hist, 90)
        del gabor
        del phase
        cv2.imwrite('temp.jpg', mag)
        input_img_gabor3.append(cv2.imread('temp.jpg'))
        os.remove('temp.jpg')

        gabor, mag, phase = gabor_v2(code_hist, 135)
        del gabor
        del phase
        cv2.imwrite('temp.jpg', mag)
        input_img_gabor4.append(cv2.imread('temp.jpg'))
        os.remove('temp.jpg')

        output_label.append(label)
        input_fd.append(fd_tuple)
        input_iris_code.append(code_hist)

    input_img_gabor1 = np.array(input_img_gabor1)
    input_img_gabor2 = np.array(input_img_gabor2)
    input_img_gabor3 = np.array(input_img_gabor3)
    input_img_gabor4 = np.array(input_img_gabor4)
    input_fd = np.array(input_fd)
    input_iris_code = np.array(input_iris_code)
    output_label = np.array(output_label)

    # return input_img_gabor, input_fd, label

    np.save('input_img_gabor1.npy', input_img_gabor1)
    np.save('input_img_gabor2.npy', input_img_gabor2)
    np.save('input_img_gabor3.npy', input_img_gabor3)
    np.save('input_img_gabor4.npy', input_img_gabor4)
    np.save('input_fd.npy', input_fd)
    np.save('input_iris.npy', input_iris_code)
    np.save('label.npy', output_label)

def create_gabor_img(df):

    for row in df.itertuples(index=False):

        filename = row.Img_Path.split('.')[0]
        filename = filename.split('/')[-1]
        print(filename)

        npz_file = np.load(row.NPZ_Path, allow_pickle=True)
        code_hist = npz_file['code_hist_eq']

        for theta in [0, 45, 90, 135]:
            gaborv2_and_create(code_hist, theta, row.Class_num, row.Class_lr, filename)

def add_fd_2_csv():

    op = open("db/CASIA-Iris-Thousand.csv", "r")
    dt = csv.DictReader(op)
    # print(dt)
    up_dt = []
    for r in dt:
        # print(r)

        npz_path = r['NPZ_Path']
        
        try:
            npz_file = np.load(npz_path, allow_pickle=True)

            dict_fd = npz_file['fd_hist_eq'].item()

            row = {'Index': r['Index'],
                'Label': r['Label'],
                'Class_num': r['Class_num'],
                'Class_lr': r['Class_lr'],
                'fd_box_counting': dict_fd['box_counting_original']['fd'],
                'fd_temporal_sampling': dict_fd['temporal_sampling']['fd'],
                'fd_corr_sum': dict_fd['corr_sum']['fd'],
                'Img_Path': r['AbsPath'],
                'NPZ_Path': r['NPZ_Path']}
        
        except:
            row = {'Index': r['Index'],
                'Label': r['Label'],
                'Class_num': r['Class_num'],
                'Class_lr': r['Class_lr'],
                'fd_box_counting': 0,
                'fd_temporal_sampling': 0,
                'fd_corr_sum': 0,
                'Img_Path': r['AbsPath'],
                'NPZ_Path': r['NPZ_Path']}
            
        up_dt.append(row)
    # print(up_dt)
    op.close()
    op = open("db/CASIA-Iris-Thousandv2.csv", "w", newline='')
    headers = ['Index', 'Label', 'Class_num', 'Class_lr', 'fd_box_counting', 'fd_temporal_sampling', 'fd_corr_sum', 'Img_Path', 'NPZ_Path']
    data = csv.DictWriter(op, delimiter=',', fieldnames=headers)
    data.writerow(dict((heads, heads) for heads in headers))
    data.writerows(up_dt)

    op.close()

if __name__ == "__main__":

    main()