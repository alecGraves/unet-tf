'''
Train a network on a dataset. Looks in hardcoded directory for npz files which are train/test pairs.
For every x, y data pair, save a npz in the data_dir.
ex.
for i in range(len(images)):
    np.save('data_dir/data'+str(i), x=images[i], y=labels[i])

model weights are saved in ../training/weights
'''
import os
import time
import numpy as np
from skimage.transform import rotate
import tensorflow as tf
join = os.path.join

from unet import create_unet
from losses import crossentropy_loss, iou_loss
from data import preprocess_label

### set these parameteres ###
#base_folder = '/media/xena/Q/data/'
base_folder = '/home/marvin'
data_dir = base_folder + 'road_detector/train'
training_split = .97
unet_depth = 3
batch_size = 8
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
    print('loading')
    batch = [[], []]
    idx = np.random.randint(0, len(datafiles)-1)
    sample = np.load(join(data_dir, datafiles[idx]))
    data = sample['x']
    data = np.add(data, 1.0)
    label = sample['y']
    label = np.expand_dims(label[:,:,0], -1)
    # label = preprocess_label(label) # oops TODO: do this when saving data.
    # sample = [np.random.randn(1300, 1300, 7), np.random.randn(1300, 1300, 2)]
    while len(batch[0]) < batch_size:
        # print(np.max(label))
        # exit()
        x_idx = np.random.randint(0, data.shape[0]-shape[0])
        y_idx = np.random.randint(0, data.shape[1]-shape[1])
        batch_data = data[x_idx:x_idx+shape[0], y_idx:y_idx+shape[1]]
        batch_label = label[x_idx:x_idx+shape[0], y_idx:y_idx+shape[1]]
        if np.random.random() < flip_prob:
            if np.random.random() < 0.5:
                batch_data = batch_data[:,::-1,...]
                batch_label = batch_label[:,::-1,...]
            else:
                batch_data = batch_data[::-1,:,...]
                batch_label = batch_label[::-1,:,...]
        # if np.random.random() < rotate_prob: # dont use, causes white boarders
        #     angle = np.random.randint(rotate_angle[0], rotate_angle[1])
        #     batch_data = rotate(batch_data, angle)
        #     batch_label = rotate(batch_label, angle)
        batch[0].append(batch_data)
        batch[1].append(batch_label)
    print('loading complete')
    return [np.array(b) for b in batch]

### Building the Graph ###
def connect_loss(batch, net):
    '''connects the loss function to the graph, 
    returns loss tensor'''
    with tf.name_scope('loss'):
        label = tf.placeholder(tf.float32, shape=batch[1].shape, name='label')
        loss = iou_loss(label, net, batch[1].shape[-1])
    return loss

def connect_optimizer(cost, learning_rate=1e-5, train_vars=None):
    '''connects optimizer to the graph,
    give it the cost tensor to minimize.
    returns the training op.'''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(cost, var_list=train_vars, colocate_gradients_with_ops=True)
    return train_op

def check_path(save_path=None):
    '''makes sure output path exists'''
    if save_path is not None and not(os.path.exists(save_path)):
        os.makedirs(save_path)

if __name__ == "__main__":
    ### Training  ###
    with tf.Session() as sess:
        with tf.device('/gpu:0'):
            batch = load_batch(train_files)

            # build the network graph
            net = create_unet(in_shape=batch[0].shape, out_channels=batch[1][0].shape[-1], name=network_name)

        with tf.device('/cpu:0'):
            saver = tf.train.Saver()
            check_path(save_path)

        with tf.device('/gpu:0'):
            loss = connect_loss(batch, net)
            train_op = connect_optimizer(loss)


        sess.run(tf.global_variables_initializer())

        losses = {'train' : [], 'val' : [], 'avgVal' : 1000000000}
        for i in range(num_epochs*len(train_files)):
            # train on a batch
            batch = load_batch(train_files)
            print('run')
            _, score = sess.run([train_op, loss], feed_dict={network_name+'/input:0':batch[0], 'loss/label:0':batch[1]})
            losses['train'].append(score)
            print('run complete')
            # test on a validation batch
            if i % 10 == 0:
                batch = load_batch(val_files)
                score = sess.run(loss, feed_dict={network_name+'/input:0':batch[0], 'loss/label:0':batch[1]})
                losses['val'].append(score)

            if i > 100:
                avg_val = sum(losses['val'][-10:])
                if avg_val < losses['avgVal']:
                    losses['avgVal'] = sum(losses['val'][-10:])
                    print('Saving model best')
                    saver.save(sess, join(save_path, 'trained-'+network_name+'best'), write_meta_graph=False)

            if ((i % (len(train_files)//batch_size//10)) == 0): # every 10th of an epoch
                print('Saving model regular')
                saver.save(sess, join(save_path, 'trained-'+network_name), global_step=i, write_meta_graph=False)
            
            if i > 10:
                print("Train Loss: {} | Val Loss: {}".format(losses['train'][-1], losses['val'][-1]))
            else:
                print("Train Loss: {} | Val Loss: Too Early".format(losses['train'][-1]))

