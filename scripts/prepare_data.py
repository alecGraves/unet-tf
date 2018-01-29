## this preparation tool is created for my data
# it could be used as inspiration for other datasets
# copyright 2018 shadySource
# MIT Liscense

import os
from os.path import join
import numpy as np
from PIL import Image

input_data_dir = '/home/marvin/road_detector/train/AOI_4_Shanghai_Roads_Test_Public'
label_data_dir = '/home/marvin/road_detector/train/AOI_4_Shanghai_Roads_Test_Public/MUL-PanSharpen'
output_data_dir = '/home/marvin/road_detector/train/data'

# for my files, the last integer in the filename is the data number.
name_key = lambda x : [int(s) for s in x.split() if s.isdigit()][-1]


if not os.path.exists(output_data_dir):
    os.makedirs(output_data_dir)

input_files = os.listdir(input_data_dir, sort=name_key)
label_files = os.listdir(label_data_dir, sort=name_key)

print(input_files[0])
print(label_files[0])

# for i in range(len(input_files)):
#     input_file = input_files[i]
#     input_file = input_file[input_file.find('img'):]
#     image_num = [int(s) for s in input_file.split() if s.isdigit()][-1]