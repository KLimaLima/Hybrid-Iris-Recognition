import numpy as np
import cv2
import matplotlib.pyplot as plt

def mask(img):
    # create a polygons using all outer corners of the ROI
    external_poly = np.array( [[[0,0],[13,45],[1,1],[23,0],[25,74], [194,60], [14,60]]], dtype=np.int32 )
    # im = cv2.imread("original.png", 1)
    cv2.fillPoly( img , external_poly, (0,0,0) )
    cv2.imwrite("output.jpg", img) 