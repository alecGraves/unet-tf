## this preparation tool is created for my data
# it could be used as inspiration for other datasets
# copyright 2018 shadySource
# MIT Liscense

import os
from os.path import join
import cv2
import numpy as np

def preprocess_image(x, mean=None, std=None):
    x = x.astype(np.float64)
    channel_max = np.max(np.max(x, axis=0), axis=0).reshape(1, 1, x.shape[-1])
    channel_max[channel_max < .1] = 1000 
    x = np.divide(x, channel_max)
    if mean is not None and std is not None:
        x = (x - np.array(mean).reshape(1,1,x.shape[-1])) / np.array(std).reshape(1,1,x.shape[-1]) 
    else:
        x = np.subtract(x, 0.5)
        x = np.multiply(x, 2.0)
    return x.astype(np.float16)

def unprocess_image(x, mean=None, std=None, bits=8):
    if mean is not None and std is not None:
        x = (x + np.array(mean).reshape(1,1,x.shape[-1])) * np.array(std).reshape(1,1,x.shape[-1])
    else:
        x = np.divide(x, 2.0)
        x = np.add(x, 0.5)
    x = np.multiply(x, 2**bits-1)
    x = x.astype(np.uint8)
    return x

def preprocess_label(x):
    x = x.astype(np.float16)
    x = np.divide(x, 255)
    return x

def unprocess_label(x):
    x = np.multiply(x, 255)
    x = x.astype(np.uint8)
    return x

if __name__ == "__main__":
    # for my files, the last integer in the filename is the data image number.
    name_key = lambda x : int(''.join([s for s in x[x.find('img'):] if s.isdigit()]))

    output_data_dir = '/home/marvin/road_detector/train'
    input_data = '/home/marvin/road_detector/unzipped/inputs'
    labels_data = '/home/marvin/road_detector/unzipped/labels/'

    for _dir in os.listdir(input_data):

        input_data_dirs = [join(input_data, _dir, 'MUL-PanSharpen'),
                            join(input_data,_dir,'RGB-PanSharpen')]
        label_data_dirs = [join(labels_data, _dir[:-12]+ '_Masks'),
                            join(labels_data, _dir[:-12]+ '_Roads_Nodes')]

        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)

        input_file_lists = [os.listdir(input_data_dir) for input_data_dir in input_data_dirs]
        input_keys_lists = [[name_key(f) for f in input_files] for input_files in input_file_lists]

        label_files_lists = [os.listdir(label_data_dir) for label_data_dir in label_data_dirs]
        label_keys_lists = [[name_key(f) for f in label_files] for label_files in label_files_lists]

        x = []
        y = []
        dataset_name = os.path.basename(label_data_dirs[0])
        for j, label_file in enumerate(label_files_lists[0]):
            img_number = label_keys_lists[0][j]
            try:
                input_file_idx = [input_keys.index(img_number) for input_keys in input_keys_lists]
            except ValueError: # input image_number is not in the list of files
                continue
            input_files = [input_file_lists[i][input_file_idx[i]] for i in range(len(input_file_idx))]
            image = [cv2.imread(join(input_data_dirs[i], input_files[i]), cv2.IMREAD_UNCHANGED) for i in range(len(input_file_idx))]
            
            image = [preprocess_image(x_i) for x_i in image]
            image = np.concatenate(image, axis=-1)

            # Show image layers and exit:
            # image = unprocess_image(image, 8)
            # for i in range(image.shape[-1]):
                # cv2.imshow('image', image[:, :, i])
                # cv2.waitKey(2000)
                # print(np.mean(image[:, :, i]))
                # print(np.max(image[:, :, i]))
                # print(np.min(image[:, :, i]))
            # exit()

            label_file_idx = [label_keys.index(img_number) for label_keys in label_keys_lists]
            label_files = [label_files_lists[i][label_file_idx[i]] for i in range(len(label_file_idx))]
            labels = [np.expand_dims(cv2.imread(join(label_data_dirs[i], label_files[i]), cv2.IMREAD_GRAYSCALE), axis=-1) for i in range(len(label_file_idx))]
            labels = np.concatenate(labels, axis=-1)
            labels = preprocess_label(labels)
            # for i in range(labels.shape[-1]):
            #     # cv2.imshow('label', labels)
            #     # cv2.waitKey(10000)
            #     print(np.mean(labels[:, :, i]))
            #     print(np.max(labels[:, :, i]))
            #     print(np.min(labels[:, :, i]))
            # exit()

            x.append(image)
            y.append(labels)
            if j%100==0 and j > 0: # about 3 gigs uncompressed
                x = np.array(x)
                y = np.array(y)
                print('x', x.shape, 'y', y.shape)
                print('saving as', join(output_data_dir, dataset_name + str(img_number)), '...')
                np.savez_compressed(join(output_data_dir, dataset_name + str(img_number)), x=x, y=y)
                x = []
                y = []
        # save anything left ovewr
        x = np.array(x)
        y = np.array(y)
        print('x', x.shape, 'y', y.shape)
        print('saving as', join(output_data_dir, dataset_name + str(img_number)), '...')
        np.savez_compressed(join(output_data_dir, dataset_name + str(img_number)), x=x, y=y)
        print('!!SAVING COMPLETE!!')