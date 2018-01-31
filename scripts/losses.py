import tensorflow as tf

def iou_loss(y_true, y_pred, num_classes=2):
    '''
    computes the IOU loss for a group of labels
    '''
    print((y_pred.get_shape()).as_list())
    # flatten labels:
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # calculate intersection and union
    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred))
    union = tf.reduce_sum(tf.subtract(tf.add(y_true, y_pred), tf.multiply(y_true, y_pred)))

    # calculate iou loss
    iou = tf.divide(intersection, union)
    loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), iou)

    return loss

def crossentropy_loss(labels, logits, num_classes=2):
    logits=tf.reshape(logits, (-1,num_classes))
    trn_labels=tf.reshape(labels, (-1,num_classes))
    cross_entropy=tf.losses.sigmoid_cross_entropy(trn_labels, logits)
    loss=tf.reduce_mean(cross_entropy, name='x_ent_mean')
    return loss