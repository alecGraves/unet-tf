## this preparation tool is created for my data
# it could be used as inspiration for other datasets
# copyright 2018 shadySource
# MIT Liscense

import os
from os.path import join
import cv2
import numpy as np

input_data_dir = 'D:\\data\\road_detector\\zipped_files\\AOI_4_Shanghai_Roads_Train\\MUL-PanSharpen'
# input_data_dir = 'D:\\data\\road_detector\\zipped_files\\AOI_4_Shanghai_Roads_Train\\RGB-PanSharpen'
label_data_dir = 'D:\\data\\road_detector\\image_tools\\AOI_4_Shanghai_Masks'
output_data_dir = 'D:\\data\\road_detector\\image_data'

# for my files, the last integer in the filename is the data image number.
name_key = lambda x : int(''.join([s for s in x[x.find('img'):] if s.isdigit()]))


if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

input_files = os.listdir(input_data_dir)
input_keys = [name_key(f) for f in input_files]
label_files = os.listdir(label_data_dir)

# my files are fucked, so sorting will not work.
# input_files.sort(key=name_key)
# label_files.sort(key=name_key)

# print(len(input_files))
# print(len(label_files))

# print(input_files[-2])
# print(label_files[-2])

def preprocess_image(x, bits=8):
    x = x.astype(np.float32)
    x = np.divide(x, 2**bits) 
    x = np.subtract(x, 1.0) 
    x = np.multiply(x, 2.0) 
    return x

def unprocess_image(x, bits=8):
    x = x.astype(np.float32)
    x = np.divide(x, 2.0)
    x = np.add(x, 1.0)
    x = np.multiply(x, 2**bits)
    if bits == 8:
        x = x.astype(np.uint8)
    elif bits == 16:
        x = x.astype(np.uint16)
    else:
        ValueError('bits must be 8 or 16')
    return x

for label_file in label_files:
    input_file_idx = input_keys.index(name_key(label_file))
    input_file = input_files[input_file_idx]
    assert os.path.exists(join(input_data_dir, input_file))
    x = cv2.imread(join(input_data_dir, input_file), cv2.IMREAD_UNCHANGED)
    if x.dtype == 'uint8':
        bits = 8
    elif x.dtype == 'uint16':
        bits = 16
    x *= np.uint16(2**bits/np.max(np.max(x, axis=0), axis=0)) # expand image to use full channel range
    x = preprocess_image(x, bits=bits)
    
    # Show image layers and exit:
    x = unprocess_image(x, 8)
    for i in range(x.shape[-1]):
        cv2.imshow('image', x[:, :, i])
        cv2.waitKey(2000)
    exit()

    y = cv2.imread(join(label_data_dir, label_file), cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('label', y)
    # cv2.waitKey(10000)
    # exit()
    print(x.shape, y.shape)
    exit()