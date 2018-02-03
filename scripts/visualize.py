import os
import tensorflow as tf
import numpy as np
import cv2

from model import UNet
# from unet import create_unet
from data import unprocess_label, unprocess_image, data_generator

sess = tf.Session()

val_dir = 'D:\\data\\road_detector\\val'
# val_dir = 'D:\\data\\road_detector\\train2'
load_from_checkpoint = '..\\training\\weights\\model-2799'
img_shape = [512, 512]
batch_size = 12
test_generator = data_generator(val_dir, batch_size=batch_size, shape=img_shape, flip_prob=0)

num_test_samples = 100

with tf.name_scope('unet'):
    model = UNet().create_model(img_shape=img_shape+[3], num_class=1)
    img = model.input
    pred = model.output
    # img, pred = create_unet(in_shape=img_shape+[7], out_channels=1, depth=5, training=True)


saver = tf.train.Saver()

with sess.as_default():
    # restore from a checkpoint if exists
    # the name_scope can not change 
    if load_from_checkpoint != '':
        try:
            saver.restore(sess,load_from_checkpoint)
            print ('--> load from checkpoint '+load_from_checkpoint)
        except:
                print ('unable to load checkpoint ...' + str(e))
        
    for ti in range(num_test_samples//batch_size):
        # tensorflow wants a different tensor order
        x_batch, y_batch = next(test_generator)
        feed_dict = { 
                        img: x_batch,
                    }
        pred_logits = sess.run(pred, feed_dict=feed_dict)
        pred_logits = 1/(1+np.exp(-1*np.array(pred_logits)))

        print(pred_logits.shape)
        print(np.max(pred_logits))
        x_batch = np.array(x_batch)
        print(np.argmax(pred_logits))
        print(np.mean(pred_logits))
        print(x_batch.shape)
        # x_batch = unprocess_image(x_batch)
        y_batch = unprocess_label(y_batch)
        # pred_logits = pred_logits > 0.1
        pred_logits = unprocess_label(pred_logits)

        for i in range(x_batch.shape[0]):
            image = unprocess_image(x_batch[i])
            # for j in range(x_batch.shape[-1]):
            #     cv2.imshow()
            # for j in range(pred_logits.shape[-1]):
            cv2.imshow('image', image[..., -3:])
            cv2.imshow('pred',pred_logits[i][...,0])
            cv2.imshow('label',y_batch[i][...,0])
            cv2.waitKey(1000)
