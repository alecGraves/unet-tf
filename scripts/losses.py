import tensorflow as tf

def iou_loss(y_true, y_pred):
    '''
    computes the IOU loss for a group of labels
    '''
    # flatten labels:
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # calculate intersection and union
    intersection = tf.multiply(y_true, y_pred)
    union = tf.subtract(tf.add(y_true, y_pred), intersection)

    # calculate iou loss
    iou = tf.divide(tf.reduce_sum(intersection), tf.reduce_sum(union))
    loss = tf.subtract(tf.constant(1.0, dtype=tf.float32), iou)

    return loss