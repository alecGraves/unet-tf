'''
Train a network on a dataset. Looks in hardcoded directory for npz files which are train/test pairs.
For every x, y data pair, save a npz in the data_dir.
ex.
for i in range(len(images)):
    np.save('data_dir/data'+str(i), x=images[i], y=labels[i])

model weights are saved in ../training/weights
'''
import os
import numpy as np
import tensorflow as tf
join = os.path.join

from unet import create_unet
from losses import iou_loss

### set these parameteres ###
train_data_dir = '/home/$USER/Desktop/data/train'
val_data_dir = '/home/$USER/Desktop/data/val'
unet_depth = 3
batch_size = 16
num_epochs = 1
network_name = 'net' # used for training and scope

### Data Management ###
# read the data files
train_files = os.listdir(train_data_dir)
val_files = os.listdir(val_data_dir)

# function to load the batches
def load_batch(datafiles):
    "Loads a batch of data from the data directory. See description above."
    batch = [[], []]
    while len(data) < batch_size:
        idx = np.random.randint(0, len(datafiles)-1)
        sample = np.load(join(data_dir, datafiles[idx]))
        batch[0].append(sample['x'])
        batch[1].append(sample['y'])
    return np.array(batch)

# get shape info for training graph
batch = load_batch()

### Building the Graph ###
# build the network graph
net = create_unet(in_shape=batch[0].shape, out_channels=batch[1][0].shape[-1], name=network_name)
net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, network_name)
assert(len(net_vars) > 0) # assert valid variables to save

def connect_loss(net):
    '''connects the loss function to the graph, 
    returns loss tensor'''
    with tf.name_scope('loss'):
        label = tf.placeholder(tf.float32, shape=batch[1].shape, name='label')
        loss = iou_loss(label, net)
    return loss

def connect_optimizer(cost, learning_rate=0.001):
    '''connects optimizer to the graph,
    give it the cost tensor to minimize.
    returns the training op.'''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(cost, var_list=trainable)

def connect_saver(save_path=None, save_vars=None):
    '''connects saver to the graph.
    Returns the tf.train.saver object'''
    with tf.name_scope('saving'):
        if save_path is None:
            save_path =  join('..', 'training', 'weights') # default saving path
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        weights_saver = tf.train.Saver(var_list=save_vars)

### Training  ###
with tf.Session() as sess:
    step = 0
    fine_tuned_saver.save(sess, join(save_path, 'trained-'+network_name, global_step=step)) # filename = 'trained-unet-0'
    step += 1

    converged = False
    while not converged:
        train()
