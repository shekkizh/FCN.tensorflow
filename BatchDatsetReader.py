#!/usr/bin/env python
# encoding: utf-8
"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc

class BatchDatset:
    files = []
    images = []
    annotations = []
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list    # 文件列表
        self.image_options = image_options
        self._read_images()

    def _read_images(self):
        self.__channels = True
        # 扫描files字典中所有image 图片全路径
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self.__channels = False
        # 扫描files字典中所有annotation 图片全路径, 根据文件全路径读取图像，并将annotation从三个维度扩充为四维度，第四个维度值为1
        self.annotations = np.array(
            [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])
        print (self.images.shape)
        print (self.annotations.shape)

    def _transform(self, filename):
        # 使用模块scipy.misc读取文件图片
        image = misc.imread(filename)
        # 如果是训练图片并且是单通道的，即维度为(h,w)，就重复三次变成3通道
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])
        # False 是从字典获取resize值属性不存在返回的默认值
        if self.image_options.get("resize", False) and self.image_options["resize"]:
            # 获取需要调整的图片大小值
            resize_size = int(self.image_options["resize_size"])
            # misc.imresize调整图片大小，使用最近邻插值法resize图片
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)  # 返回已经resize的图片

    def get_records(self):
        """
        :return:图片和标签
        """
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        '''

        :param offset:
        :return:
        '''
        self.batch_offset = offset

    def next_batch(self, batch_size):
        # 将batch偏移量赋值给start
        start = self.batch_offset
        # next_batch()方法每调用一次，batch_offset的值偏移一个batch_size大小
        self.batch_offset += batch_size
        # 如果出现总偏移量大于训练集数量
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data--接下来的每次迭代都先生成一个索引序列，按照这个序列重排数据集，相当于数据清洗
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        # 结束位置索引
        end = self.batch_offset
        # 按照索引得到训练图片及对应标签
        return self.images[start:end], self.annotations[start:end]

    def next_batch_with_name(self, batch_size):
        # 将batch偏移量赋值给start
        start = self.batch_offset
        # next_batch()方法每调用一次，batch_offset的值偏移一个batch_size大小
        self.batch_offset += batch_size
        # 结束位置索引
        end = self.batch_offset
        # 按照索引得到训练图片及对应标签
        return self.images[start:end], self.annotations[start:end], self.files[start:end]

    def get_random_batch(self, batch_size):
        '''
        # 按照一个batch_size一个块  进行对所有图片总数进行随机操作， 相当于洗牌工作
        :param batch_size: 指定的batch_size块大小
        :return:batch_size个大小的图片和对应标签
        '''
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]
