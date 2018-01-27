import tensorflow as tf

def create_unet(in_shape=[1, 288, 288, 3], depth=5, training=True, name='UNET'):
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
        net = tf.placeholder(tf.float32, shape=in_shape, name='input')

        # unet down
        down_outputs = []
        for i in range(depth):
            with tf.name_scope('down_'+str(i)):
                # downsample image if not the input
                if i > 0:
                    net = max_pool2d(net, kernel_size=(3, 3), name='max_pool')
                
                # conv block 1
                net = tf.nn.contrib.layers.conv2d(net,num_outputs=32*(2**i), kernel_size=(3, 3), activation_fn=None, name='conv_1')
                net = tf.nn.selu(net, name='act_1')
                net = tf.nn.fused_batch_norm(net, is_training=training, name='bn_1')
                
                # conv block 2
                net = tf.nn.contrib.layers.conv2d(net,num_outputs=32*(2**i), kernel_size=(3, 3), activation_fn=None, name='conv_2')
                net = tf.nn.selu(net, name='act_2')
                net = tf.nn.fused_batch_norm(net, is_training=training, name='bn_2')

                # save the output
                down_outputs.append(net)

        # unet up
        for i in range(depth-1):
            with tf.name_scope('up_'+str(i)):
                # upsample the ouput
                _, h, w, _ = net.get_shape().as_list()
                net = tf.image.resize_bicubic(net, size=(2*h, 2*w),name='upsample_2x')

                # concatenate and dropout
                net = tf.concat([net, down_outputs.pop()], axis=-1, name='concat')
                net = tf.contrib.layers.dropout(net, keep_prob=0.6, is_training=training, name='dropout')
                
                # conv block 1
                net = tf.nn.contrib.layers.conv2d(net,num_outputs=32*(2**(depth-(i+2))), kernel_size=(3, 3), activation_fn=None, name='conv_1')
                net = tf.nn.selu(net, name='act_1')

                # conv block 2
                net = tf.nn.contrib.layers.conv2d(net,num_outputs=32*(2**(depth-(i+2))), kernel_size=(3, 3), activation_fn=None, name='conv_2')
                net = tf.nn.selu(net, name='act_2')

        # final output layer, 1 class
        net = tf.nn.contrib.layers.conv2d(net, num_outputs=1, kernel_size=(1, 1), activation_fn=None, name='output')

    return net