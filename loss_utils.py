#!/usr/bin/env python
# encoding: utf-8
'''
@author: lele Ye
@contact: 1750112338@qq.com
@software: pycharm 2018.2
@file: loss_utils.py
@time: 2019/5/27 16:32
@desc:
'''
import tensorflow as tf
from tensorflow.python.ops import array_ops

# log Loss
def log_loss(y_true, y_pred, num_classes):
    with tf.name_scope("softmax_loss"):
        y_pred = tf.reshape(y_pred, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))

        sigmoid = tf.nn.sigmoid(y_pred) + epsilon

        corss_entropy = -tf.reduce_sum(y_true * tf.log(sigmoid), reduction_indices=[1])

        corss_entropy_mean = tf.reduce_mean(corss_entropy, name="xentropy_mean")
        tf.add_to_collection("losses", corss_entropy_mean)
        loss = tf.add_n(tf.get_collection("losses"), name="total_loss")
    return loss

# softmax+cross entropy loss，比如fcn和u-net
def cross_entroy_loss(y_true, y_pred, num_classes):
    '''
    calculate the loss from the y_true and the y_pred.
    :param y_true: tensor,float -[batch-size,width,heitht,num_classes].
        use vgg_fcn.upscore as y_true.
    :param y_pred: y_pred tensor,int32 -[batch_size, width,height, num_classes]
        the ground truth of data
    :param num_classes:
    :return: Loss tensor of type float
    '''
    with tf.name_scope("softmax_loss"):
        y_pred = tf.reshape(y_pred, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        y_true = tf.to_float(tf.reshape(y_true, (-1, num_classes)))

        softmax = tf.nn.softmax(y_pred) + epsilon

        corss_entropy = -tf.reduce_sum(y_true * tf.log(softmax), reduction_indices=[1])

        corss_entropy_mean = tf.reduce_mean(corss_entropy, name="xentropy_mean")
        tf.add_to_collection("losses", corss_entropy_mean)
        loss = tf.add_n(tf.get_collection("losses"), name="total_loss")
    return loss

# sigmoid+dice loss, 比如v-net，只适合二分类，直接优化评价指标


# 第一的加权版本，比如segnet

def dice_coe(y_pred, y_true, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background),
             dice = ```smooth/(small_value + smooth)``, then if smooth is very small,
             dice close to 0 (even the image values lower than the threshold), so in this case,
             higher smooth can have a higher dice.
    """
    inse = tf.reduce_sum(y_pred * y_true, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(y_pred * y_pred, axis=axis)
        r = tf.reduce_sum(y_true * y_true, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(y_pred, axis=axis)
        r = tf.reduce_sum(y_true, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice

def dice_coef_loss(y_true, y_pred):
    1 - dice_coef(y_true, y_pred, smooth=1)


def focal_loss(y_pred, y_true, weights=None, alpha=0.25, gamma=2):
    r"""
    code form:https://github.com/ailias/Focal-Loss-implement-on-Tensorflow
    Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = y_true.
    Args:
     y_pred: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     y_true: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(y_pred)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # y_true > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(y_true > zeros, y_true - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # y_true > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(y_true > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def sparse_softmax_cross_entropy_with_logits(y_true, y_pred):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred,
                                                                          labels=tf.squeeze(y_true, squeeze_dims=[3]),
                                                                          name="entropy"))