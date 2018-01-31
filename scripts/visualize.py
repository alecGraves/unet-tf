import tensorflow as tf
import numpy as np
import cv2
from data import unprocess_label
from unet import create_unet
from train import val_files, load_batch

with tf.Session() as sess:
    batch = load_batch(val_files)

    # build the network graph
    net_name = 'net'
    net = create_unet(in_shape=batch[0].shape, out_channels=batch[1][0].shape[-1], name=net_name, training=False)
    net = tf.nn.sigmoid(net)
    saver = tf.train.Saver()
    saver.restore(sess, "C:\\Users\\alec0\\Documents\\GitHub\\unet-tf\\training\\weights\\trained-net-1092")

    output = sess.run(net, feed_dict={net_name+'/input:0':batch[0]})
    print(np.max(output))
    print(np.max(batch[0]))
    print(np.max(batch[1]))
    output = unprocess_label(output)
    batch[1] = unprocess_label(batch[1])
    for i in range(output.shape[0]):
        for j in range(output.shape[-1]):
            cv2.imshow('pred',output[i][:,:,j])
            cv2.imshow('label',batch[1][i][:,:,j])
            cv2.waitKey(2000)

