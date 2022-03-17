#!/usr/bin/env python
# encoding: utf-8
# Utils used with tensorflow implemetation
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import scipy.io


# 获取VGG预训练模型
def get_model_data(dir_path, model_url):
    '''
    下载预训练模型文件，并返回已经加载的模型数据
    :param dir_path: Model需要保存的路径
    :param model_url: 下载VGG19网址
    :return: 预训练模型
    '''
    maybe_download_and_extract(dir_path, model_url)  # 判断文件目录和文件是否存在， 不存在则下载
    filename = model_url.split("/")[-1]  # 将url按/切分， 取最后一个字符串作为文件名
    filepath = os.path.join(dir_path, filename)  # dir_path/filename     文件全路径
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!")
    data = scipy.io.loadmat(filepath)  # 使用io读取VGG.mat文件
    return data


def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    '''
    用于下载模型及数据文件，如果是压缩包同时支持解压缩
    :param dir_path: 预训练Model或者数据集需要存放的文件路径
    :param url_name: 下载VGG19网址、或者是数据集的网址
    :param is_tarfile:是否是tar类型的压缩包文件
    :param is_zipfile:是否是zip类型的压缩包文件
    :return:
    '''
    if not os.path.exists(dir_path):  # 判断文件路径是否存在，如果不存在则创建此路径
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]  # 将url中 按照/切分，并取最后一个字符串 作为文件名字  或者是  os.path.basename(url_name)
    filepath = os.path.join(dir_path, filename)  # 文件路径 = dir_path/filename
    # 如果文件不存在
    if not os.path.exists(filepath):
        # 定义进度条刷新函数
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:  # 如果是tar文件， 解压缩
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:  # 如果是zip文件 解压缩
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)


def save_image(image, save_dir, name, mean=None):
    """
    Save image by unprocessing if mean given else just save
    :param mean:
    :param image:
    :param save_dir:
    :param name:
    :return:
    """
    if mean:
        image = unprocess_image(image, mean)
    misc.imsave(os.path.join(save_dir, name + ".png"), image)


def get_variable(weights, name):
    '''

    :param weights:
    :param name:
    :return:
    '''
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var


def weight_variable(shape, stddev=0.02, name=None):
    '''
    权重初始化函数
    :param shape: the shape of the weight_variable
    :param stddev: 权重变量的标准差
    :param name:
    :return:
    '''
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    '''
    根据变量名称获取偏置变量
    :param shape:偏置变量的shape，
    :param name:偏置变量的名称
    :return:
    '''
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def get_tensor_size(tensor):
    '''
    获取张量维度
    :param tensor:
    :return:
    '''
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def conv2d_basic(x, W, bias):
    '''
    单步幅香草卷积
    :param x:卷积输入特征图
    :param W:卷积核
    :param bias:偏置
    :return:
    '''
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)


def conv2d_strided(x, W, b):
    '''
    2步幅香草卷积
    :param x:
    :param W:
    :param b:
    :return:
    '''
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def conv2d_transpose_strided(x, W, b, output_shape=None, stride=2):
    '''
    2步幅转置卷积
    :param x:
    :param W:
    :param b:
    :param output_shape:
    :param stride:
    :return:
    '''
    # print x.get_shape()
    # print W.get_shape()
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    # print output_shape
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
    return tf.nn.bias_add(conv, b)


def leaky_relu(x, alpha=0.0, name=""):
    '''
    leaky relu激活函数
    :param x:
    :param alpha:
    :param name:
    :return:
    '''
    return tf.maximum(alpha * x, x, name)


def max_pool_2x2(x):
    '''
    最大池化函数
    :param x:
    :return:
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def avg_pool_2x2(x):
    '''
    均值池化函数
    :param x:
    :return:
    '''
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def local_response_norm(x):
    '''
    局部相应归一化函数
    :param x:
    :return:
    '''
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)


def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5):
    '''
    Code taken from http://stackoverflow.com/a/34634291/2267819
    批量归一化函数
    :param x:
    :param n_out:
    :param phase_train:
    :param scope:
    :param decay:
    :param eps:
    :return:
    '''
    with tf.variable_scope(scope):
        beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0)
                               , trainable=True)
        gamma = tf.get_variable(name='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, 0.02),
                                trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed


def process_image(image, mean_pixel):
    '''
    对图像减均值处理
    :param image:
    :param mean_pixel:
    :return:
    '''
    return image - mean_pixel


def unprocess_image(image, mean_pixel):
    '''
    图像加均值还原
    :param image:
    :param mean_pixel:
    :return:
    '''
    return image + mean_pixel


def bottleneck_unit(x, out_chan1, out_chan2, down_stride=False, up_stride=False, name=None):
    """
    Modified implementation from github ry?!
    """

    def conv_transpose(tensor, out_channel, shape, strides, name=None):
        out_shape = tensor.get_shape().as_list()
        in_channel = out_shape[-1]
        kernel = weight_variable([shape, shape, out_channel, in_channel], name=name)
        shape[-1] = out_channel
        return tf.nn.conv2d_transpose(x, kernel, output_shape=out_shape, strides=[1, strides, strides, 1],
                                      padding='SAME', name='conv_transpose')

    def conv(tensor, out_chans, shape, strides, name=None):
        in_channel = tensor.get_shape().as_list()[-1]
        kernel = weight_variable([shape, shape, in_channel, out_chans], name=name)
        return tf.nn.conv2d(x, kernel, strides=[1, strides, strides, 1], padding='SAME', name='conv')

    def bn(tensor, name=None):
        """
        :param tensor: 4D tensor input
        :param name: name of the operation
        :return: local response normalized tensor - not using batch normalization :(
        """
        return tf.nn.lrn(tensor, depth_radius=5, bias=2, alpha=1e-4, beta=0.75, name=name)

    in_chans = x.get_shape().as_list()[3]

    if down_stride or up_stride:
        first_stride = 2
    else:
        first_stride = 1

    with tf.variable_scope('res%s' % name):
        if in_chans == out_chan2:
            b1 = x
        else:
            with tf.variable_scope('branch1'):
                if up_stride:
                    b1 = conv_transpose(x, out_channel=out_chan2, shape=1, strides=first_stride,
                                        name='res%s_branch1' % name)
                else:
                    b1 = conv(x, out_chans=out_chan2, shape=1, strides=first_stride, name='res%s_branch1' % name)
                b1 = bn(b1, 'bn%s_branch1' % name)

        with tf.variable_scope('branch2a'):
            if up_stride:
                b2 = conv_transpose(x, out_channel=out_chan1, shape=1, strides=first_stride,
                                    name='res%s_branch2a' % name)
            else:
                b2 = conv(x, out_chans=out_chan1, shape=1, strides=first_stride, name='res%s_branch2a' % name)
            b2 = bn(b2, 'bn%s_branch2a' % name)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2b'):
            b2 = conv(b2, out_chans=out_chan1, shape=3, strides=1, name='res%s_branch2b' % name)
            b2 = bn(b2, 'bn%s_branch2b' % name)
            b2 = tf.nn.relu(b2, name='relu')

        with tf.variable_scope('branch2c'):
            b2 = conv(b2, out_chans=out_chan2, shape=1, strides=1, name='res%s_branch2c' % name)
            b2 = bn(b2, 'bn%s_branch2c' % name)

        x = b1 + b2
        return tf.nn.relu(x, name='relu')


def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad)
