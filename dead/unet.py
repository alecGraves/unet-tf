import tensorflow as tf

def create_unet(in_shape=[256, 256, 3], out_channels=1, depth=5, training=True, name='UNET'):
    '''
    Creates a UNET model.
    # Params:
        in_shape = [batch_size, height, width, channels]
        depth = number of downsample blocks
        training = whether or not model is training (for b.n. and dropout)
        name = name to use for scope of model and all variables
    # Returns:
        model logits
    '''
    with tf.name_scope(name):
        inputs = tf.placeholder(tf.float32, shape=[None]+in_shape, name='input')
        net = inputs

        # unet down
        down_outputs = []
        for i in range(depth):
            with tf.name_scope('down_'+str(i)):
                # downsample image if not the input
                if i > 0:
                    net = tf.contrib.layers.max_pool2d(net, kernel_size=(3, 3), padding='SAME')
                
                # conv block 1
                net = tf.contrib.layers.conv2d(net,num_outputs=32*(2**i), kernel_size=(3, 3), activation_fn=None)
                net = tf.nn.elu(net, name='act_1')
                net = tf.contrib.layers.batch_norm(net, is_training=training)
                
                # conv block 2
                net = tf.contrib.layers.conv2d(net,num_outputs=32*(2**i), kernel_size=(3, 3), activation_fn=None)
                net = tf.nn.elu(net, name='act_2')
                net = tf.contrib.layers.batch_norm(net, is_training=training)

                # save the output
                down_outputs.append(net)

        # start up with the last output
        net = down_outputs.pop()

        # unet up
        for i in range(depth-1):
            with tf.name_scope('up_'+str(i)):
                # upsample the ouput
                bat, h, w, chan = net.get_shape().as_list()
                net = tf.contrib.layers.conv2d_transpose(net, chan, (2, 2), stride=(2, 2), padding='same')
                # tf.keras.layers.UpSampling2D(net, size=(2, 2))

                # concatenate and dropout
                net = tf.concat([net, down_outputs.pop()], axis=-1, name='concat')
                net = tf.contrib.layers.dropout(net, keep_prob=0.9, is_training=False)
                
                # conv block 1
                net = tf.contrib.layers.conv2d(net,num_outputs=32*(2**(depth-(i+2))), kernel_size=(3, 3), activation_fn=None)
                net = tf.nn.elu(net, name='act_1')

                # conv block 2
                net = tf.contrib.layers.conv2d(net,num_outputs=32*(2**(depth-(i+2))), kernel_size=(3, 3), activation_fn=None)
                net = tf.nn.elu(net, name='act_2')

        # final output layer, out_channels classes
        net = tf.contrib.layers.conv2d(net, num_outputs=out_channels, kernel_size=(1, 1), activation_fn=None)
        # net = tf.nn.sigmoid(net)
    return inputs, net

if __name__ == '__main__':
    # test the compilation of the model
    net = create_unet()