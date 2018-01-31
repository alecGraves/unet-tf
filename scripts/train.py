'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-19 03:06:32
 * @modify date 2017-05-19 03:06:32
 * @desc [description]
 * MIT License
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

SEED=0 # set set to allow reproducing runs
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.set_random_seed(SEED)

from model import UNet
from data import data_generator
from utils import VIS, mean_IU

# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

''' Users define data loader (with train and test) '''
img_shape = [256, 256]
batch_size = 8
epochs = 10
steps_per_epoch = 28*75 # 28 npz's averaging 75 images a piece
train_dir = ''
val_dir = ''
load_from_checkpoint = ''
checkpoint_path = os.path.join('..', 'training', 'weights')
tensorboard_path = os.path.join('..', 'training', 'logs')
train_generator = data_generator(train_dir, batch_size=batch_size, shape=img_shape, flip_prob=.4)
train_generator = data_generator(val_dir, batch_size=batch_size, shape=img_shape, flip_prob=0)

vis = VIS(save_path=checkpoint_path)

label = tf.placeholder(tf.int32, shape=[batch_size]+img_shape)

with tf.name_scope('unet'):
    model = UNet().create_model(img_shape=img_shape+[3], num_class=2)
    img = model.input
    pred = model.output

with tf.name_scope('cross_entropy'):
    cross_entropy_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred))

global_step = tf.Variable(0, name='global_step', trainable=False)
with tf.name_scope('learning_rate'):
    learning_rate = tf.train.exponential_decay(0.0001, global_step,
                                           steps_per_epoch, 0.9, staircase=True)

train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss, global_step=global_step)

''' Tensorboard visualization '''
# cleanup pervious info
if load_from_checkpoint == '':
    cf = os.listdir(checkpoint_path)
    for item in cf: 
        if 'event' in item: 
            os.remove(os.path.join(checkpoint_path, item))
# define summary for tensorboard
tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
tf.summary.scalar('learning_rate', learning_rate)
summary_merged = tf.summary.merge_all()
# define saver
train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
saver = tf.train.Saver() # must be added in the end


tot_iter = steps_per_epoch * epochs
init_op = tf.global_variables_initializer()
sess.run(init_op)

with sess.as_default():
    # restore from a checkpoint if exists
    # the name_scope can not change 
    if opt.load_from_checkpoint != '':
        try:
            saver.restore(sess, opt.load_from_checkpoint)
            print ('--> load from checkpoint '+opt.load_from_checkpoint)
        except:
                print ('unable to load checkpoint ...' + str(e))
    # debug
    start = global_step.eval()
    for it in range(start, tot_iter):
        if it % steps_per_epoch == 0 or it == start:
            
            saver.save(sess, opt.checkpoint_path+'model', global_step=global_step)
            print ('save a checkpoint at '+ opt.checkpoint_path+'model-'+str(it))
            print ('start testing {} samples...'.format(test_samples))
            for ti in range(test_samples):
                x_batch, y_batch = next(test_generator)
                # tensorflow wants a different tensor order
                feed_dict = {   
                                img: x_batch,
                                label: y_batch,
                            }
                loss, pred_logits = sess.run([cross_entropy_loss, pred], feed_dict=feed_dict)
                pred_map_batch = np.argmax(pred_logits, axis=3)
                # import pdb; pdb.set_trace()
                for pred_map, y in zip(pred_map_batch, y_batch):
                    score = vis.add_sample(pred_map, y)
            vis.compute_scores(suffix=it)
        
        x_batch, y_batch = next(train_generator)
        feed_dict = {   img: x_batch,
                        label: y_batch
                    }
        _, loss, summary, lr, pred_logits = sess.run([train_step, 
                                    cross_entropy_loss, 
                                    summary_merged,
                                    learning_rate,
                                    pred
                                    ], feed_dict=feed_dict)
        global_step.assign(it).eval()
        train_writer.add_summary(summary, it)
        
        pred_map = np.greater(pred_logits[0], 0.5)
        score, _ = mean_IU(pred_map, y_batch[0])

       
        if it % 20 == 0 : 
            print ('[iter %d, epoch %.3f]: lr=%f loss=%f, mean_IU=%f' % (it, float(it)/opt.iter_epoch, lr, loss, score))
        