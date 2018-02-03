'''
created by shadysource
MIT liscense
'''
import numpy as np

def IOU(y_true, y_pred):
    intersection = y_true * y_pred
    union = y_true + y_pred - intersection
    intersection = np.sum(np.sum(intersection, axis=0), axis=0)
    union = np.sum(np.sum(union, axis=0), axis=0)
    iou = intersection / union
    return iou

