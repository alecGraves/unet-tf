## this preparation tool is created for my data
# it could be used as inspiration for other datasets
# copyright 2018 shadySource
# MIT Liscense

import os
from os.path import join
import cv2
import numpy as np
from PIL import Image

def preprocess_image(x):
    x = x.astype(np.float64)
    channelmax = np.asarray(np.max(np.max(x, axis=0), axis=0))
    channelmax[channelmax < 1] = 1000
    x = x / channelmax
    x = x - 0.5
    x = x * 2.
    return x.astype(np.float16)

def unprocess_image(x):
    x = x / 2
    x = x + 0.
    x = x * 255
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
    reshape_size = 512

    for _dir in os.listdir(input_data):

        input_data_dirs = [join(input_data, _dir, 'RGB-PanSharpen')]#,
                            # join(input_data,_dir,'RGB-PanSharpen')]
        label_data_dirs = [join(labels_data, _dir[:-12]+ '_Masks')]#,
                            # join(labels_data, _dir[:-12]+ '_Roads_Nodes')]

        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)

        input_file_lists = [os.listdir(input_data_dir) for input_data_dir in input_data_dirs]
        input_keys_lists = [[name_key(f) for f in input_files] for input_files in input_file_lists]

        label_files_lists = [os.listdir(label_data_dir) for label_data_dir in label_data_dirs]
        label_files_lists[0].sort(key=name_key) # this one is used to get the order of files
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
            image = [np.array(Image.fromarray(unprocess_image(preprocess_image(img))).resize((reshape_size, reshape_size))) for img in image]
            image = np.concatenate(image, axis=-1)
            # print(image.shape)
            image = preprocess_image(image)
            # Image.fromarray(unprocess_image(image)).show()
            

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
            labels = [cv2.imread(join(label_data_dirs[i], label_files[i]), cv2.IMREAD_GRAYSCALE) for i in range(len(label_file_idx))]
            labels = [np.array(Image.fromarray(img).resize((reshape_size, reshape_size))) for img in labels]
            labels = np.concatenate(labels, axis=-1)
            # print(labels.shape)
            labels = np.expand_dims(preprocess_label(labels), axis=-1)
            # Image.fromarray(unprocess_label(labels)).show()
            # exit()
            # for i in range(labels.shape[-1]):
            #     # cv2.imshow('label', labels)
            #     # cv2.waitKey(10000)
            #     print(np.mean(labels[:, :, i]))
            #     print(np.max(labels[:, :, i]))
            #     print(np.min(labels[:, :, i]))
            # exit()

            # measure mean and std
            # x.append(np.mean(np.mean(image, axis=0), axis=0))
            # print('mean',np.mean(np.array(x), axis=0))
            # y.append(np.std(np.std(image, axis=0), axis=0))
            # print('std', np.mean(np.array(y), axis=0))
            # continue

            x.append(image)
            y.append(labels)
            if j%500==0 and j > 0: # about 3 gigs uncompressed
                x = np.array(x)
                y = np.array(y)
                print('x', x.shape, 'y', y.shape)
                print('saving as', join(output_data_dir, _dir + str(img_number)), '...')
                np.savez_compressed(join(output_data_dir, _dir + str(img_number)), x=x, y=y)
                x = []
                y = []
        # save anything left ovewr
        x = np.array(x)
        y = np.array(y)
        print('x', x.shape, 'y', y.shape)
        print('saving as', join(output_data_dir, _dir + str(img_number)), '...')
        np.savez_compressed(join(output_data_dir, _dir + str(img_number)), x=x, y=y)
    print('!!SAVING COMPLETE!!')

def data_generator(data_dir, batch_size=8, shape=[256, 256], flip_prob=.5):
    while True:
        for npz_file in os.listdir(data_dir):
            data = np.load(join(data_dir, npz_file))
            data_x =  data['x']
            data_y = data['y']
            data_len = data_x.shape[0]
            for i in range(data_len//batch_size):
                image, mask = ([], [])
                for j in range(batch_size):
                    data_idx = np.random.randint(0, data_len)
                    # cropping indices
                    # x_idx = np.random.randint(0, data_x.shape[1]-shape[0]) 
                    # y_idx = np.random.randint(0, data_x.shape[2]-shape[1])
                    # x = data_x[data_idx, x_idx:x_idx+shape[0], y_idx:y_idx+shape[1], :]
                    # y = data_y[data_idx, x_idx:x_idx+shape[0], y_idx:y_idx+shape[1], :]
                    x = data_x[data_idx]
                    y = data_y[data_idx]
                    if np.random.random() < flip_prob:
                        if np.random.random() < 0.5:
                            x = x[:,::-1,:]
                            y = y[:,::-1,:]
                        if np.random.random() < 0.5:
                            x = x[::-1,:,:]
                            y = y[::-1,:,:]
                    image.append(x)
                    mask.append(y)
                yield image, mask
