'''
created by shadysource
MIT license 
'''
import os
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

from model import UNet
from data import preprocess_image, unprocess_label, unprocess_image, _unprocess_image, data_generator

train_error = True # see data._unprocess_image for info.

join = os.path.join

input_dir = 'D:\\data\\road_detector\\test'
output_dir = 'D:\\data\\road_detector\\test_masks'

load_from_checkpoint = '..\\training\\models\\model-A'
input_shape = [512, 512]
full_shape = [1300, 1300]
batch_size = 12

with tf.name_scope('unet'):
    model = UNet().create_model(img_shape=input_shape+[3], num_class=1)
    img = model.input
    pred = model.output

sess = tf.Session()
saver = tf.train.Saver()

try:
    saver.restore(sess,load_from_checkpoint)
    print ('--> load from checkpoint '+load_from_checkpoint)
except:
    print ('FATAL ERROR: unable to load checkpoint ...')
    exit()

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for test_dir in os.listdir(input_dir):
    if not os.path.exists(join(output_dir, test_dir)):
        os.mkdir(join(output_dir, test_dir))
    if not os.path.exists(join(output_dir, test_dir, 'images')):
        os.mkdir(join(output_dir, test_dir, 'images'))
    if not os.path.exists(join(output_dir, test_dir, 'masks')):
        os.mkdir(join(output_dir, test_dir, 'masks'))
    image_names = os.listdir(join(input_dir, test_dir, 'RGB-PanSharpen'))
    image_paths = [join(input_dir, test_dir, 'RGB-PanSharpen', _name) for _name in image_names]

    batch = []
    batch_names = []
    for i in range(len(image_paths)-1, 0, -1):
        if len(batch) < batch_size:
            # add an image to the batch
            batch.append(cv2.imread(image_paths[i], cv2.IMREAD_UNCHANGED))
            if train_error:
                batch[-1] = np.array(Image.fromarray(_unprocess_image(preprocess_image(batch[-1]))).resize(input_shape))
            else:
                batch[-1] = np.array(Image.fromarray(unprocess_image(preprocess_image(batch[-1]))).resize(input_shape))
            batch[-1] = preprocess_image(batch[-1])
            batch_names.append(image_paths[i])

            # zero pad if last image
            if i == 0:
                while len(batch) != batch_size:
                    batch.append(np.zeros(batch[-1].shape, dtype=np.float16))

        if len(batch) == batch_size:
            # run predictions
            pred_logits = sess.run(pred, feed_dict={img:batch})
            pred_logits = 1/(1+np.exp(-1*pred_logits))
            pred_logits = unprocess_label(pred_logits)[...,0]
            for j in range(batch_size):
                # save the mask
                filename = os.path.basename(batch_names[j])
                filename = filename[:filename.find('.')] + '.jpeg'
                savepath = join(output_dir, test_dir, 'masks', filename)
                _mask = Image.fromarray(pred_logits[j]).resize(full_shape, Image.BILINEAR)
                # _mask.show()
                _mask.save(savepath)

                # save the image
                # reuse filename
                savepath = join(output_dir, test_dir, 'images', filename)
                _fullsize_img = cv2.imread(batch_names[j], cv2.IMREAD_UNCHANGED)
                _fullsize_img = Image.fromarray(unprocess_image(preprocess_image(_fullsize_img)))
                # _fullsize_img.show()
                _fullsize_img.save(savepath)

            batch = []
            batch_names = []
            
