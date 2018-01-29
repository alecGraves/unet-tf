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
from skimage.transform import rotate
import tensorflow as tf
join = os.path.join

from unet import create_unet
from losses import iou_loss

### set these parameteres ###
data_dir = 'D:\\data\\road_detector\\image_data'
training_split = .97
unet_depth = 3
batch_size = 1
num_epochs = 1
network_name = 'net' # used for training and scope
save_path = join('..', 'training', 'weights')


### Data Management ###
# read the data files
data_files = os.listdir(data_dir)
train_files = data_files[:int(len(data_files)*training_split)]
val_files = data_files[int(len(data_files)*training_split):]

# function to load the batches
def load_batch(datafiles, shape=[256, 256], flip_prob=.4, rotate_prob=.2, rotate_angle=[-45, 45]):
    """Loads a batch of data from the data directory. See description above."""
    batch = [[], []]
    idx = np.random.randint(0, len(datafiles)-1)
    sample = np.load(join(data_dir, datafiles[idx]))
    # sample = [np.random.randn(1300, 1300, 7), np.random.randn(1300, 1300, 2)]
    while len(batch[0]) < batch_size:
        data = sample['x']
        label = sample['y']
        x_idx = np.random.randint(0, data.shape[0]-shape[0])
        y_idx = np.random.randint(0, data.shape[1]-shape[1])
        if np.random.random() < flip_prob:
            if np.random.random() < 0.5:
                data = data[:,::-1,...]
                label = label[:,::-1,...]
            else:
                data = data[::-1,:,...]
                label = label[::-1,:,...]
        if np.random.random() < rotate_prob:
            angle = np.random.randint(rotate_angle[0], rotate_angle[1])
            data = rotate(data, angle)
            label = rotate(label, angle)
        batch[0].append(data[x_idx:x_idx+shape[0], y_idx:y_idx+shape[1]])
        batch[1].append(label[x_idx:x_idx+shape[0], y_idx:y_idx+shape[1]])
    return [np.array(b) for b in batch]

### Building the Graph ###
def connect_loss(batch, net):
    '''connects the loss function to the graph, 
    returns loss tensor'''
    with tf.name_scope('loss'):
        label = tf.placeholder(tf.float32, shape=batch[1].shape, name='label')
        loss = iou_loss(label, net)
    return loss

def connect_optimizer(cost, learning_rate=0.001, train_vars=None):
    '''connects optimizer to the graph,
    give it the cost tensor to minimize.
    returns the training op.'''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(cost, var_list=train_vars)
    return train_op

def connect_saver(save_path=None, save_vars=None):
    '''connects saver to the graph.
    Returns the tf.train.saver object'''
    with tf.name_scope('saving'):
        if save_path is not None and not(os.path.exists(save_path)):
            os.makedirs(save_path)
        return tf.train.Saver(var_list=save_vars)

### Training  ###
with tf.Session() as sess:
    batch = load_batch(train_files)
    print([_.shape for _ in batch])
    # build the network graph
    net = create_unet(in_shape=batch[0].shape, out_channels=batch[1][0].shape[-1], name=network_name)
    net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, network_name)
    assert(len(net_vars) > 0) # assert valid variables to save

    loss = connect_loss(batch, net)
    train_op = connect_optimizer(loss, train_vars=net_vars)

    saver = connect_saver(save_path=save_path, save_vars=net_vars)

    losses = {'train' : [], 'val' : [], 'avgVal' : 1000000000}
    for i in range(num_epochs*len(train_files)//batch_size):
        # train on a batch
        batch = load_batch(train_files)
        _, score = sess.run([train_op, loss], feed_dict={network_name+'/input:0':batch[0], net:batch[1]})
        losses['train'].append(score)

        # test on a validation batch
        if i % 10 == 0:
            batch = load_batch(val_files)
            score = sess.run(loss, feed_dict={network_name+'/input:0':batch[0], net:batch[1]})
            losses['val'].append(score)

        if i > 100:
            avg_val = sum(losses['val'][-10:])
            if avg_val < losses['avgVal']:
                losses['avgVal'] = sum(losses['val'][-10:])
                saver.save(sess, join(save_path, 'trained-'+network_name+'best'))

        if i % len(train_files)//batch_size//10 == 0: # every 10th of an epoch
            saver.save(sess, join(save_path, 'trained-'+network_name), global_step=i)
        
        if i > 10:
            print("Train Loss: {} | Val Loss: {}".format(losses['train'][-1], losses['val'][-1]))
        else:
            print("Train Loss: {} | Val Loss: Too Early".format(losses['train'][-1]))

