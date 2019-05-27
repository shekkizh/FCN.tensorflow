#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'


def read_dataset(data_dir):
    # data_dir = Data_zoo / MIT_SceneParsing /
    pickle_filename = "MITSceneParsing.pickle"
    # 文件路径  Data_zoo / MIT_SceneParsing / MITSceneParsing.pickle
    # MITSceneParsing.pickle是序列化生成的文件，意图是将训练集和验证集的图片路径、标签路径、文件名提取并绑定
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True) # 不存在文件 则下载
        SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0] # 得到数据集名称ADEChallengeData2016
        # result =   {training: [{image: 图片全路径， annotation:标签全路径， filename:图片名字}] [][]
        #            validation:[{image:图片全路径， annotation:标签全路径， filename:图片名字}] [] []}
        result = create_image_lists(os.path.join(data_dir, SceneParsing_folder))
        # 制作pickle文件
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:  # 打开pickle文件
        result = pickle.load(f) # 读取
        training_records = result['training']
        validation_records = result['validation']
        del result
    # training_records: [{image: 图片全路径， annotation:标签全路径， filename:图片名字}] [{}][{}]
    # validation_records: [{image: 图片全路径， annotation:标签全路径， filename:图片名字}] [{}][{}]
    return training_records, validation_records


def create_image_lists(image_dir):
    """
    :param image_dir:   Data_zoo / MIT_SceneParsing / ADEChallengeData2016
    :return:
    """
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    image_list = {} # 图像字典   training:[]  validation:[]

    for directory in directories: # 训练集和验证集 分别制作
        file_list = []
        image_list[directory] = []
        # Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/training/*.jpg
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
        # 加入文件列表  包含所有图片文件全路径+文件名字  如 Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/training/hi.jpg
        file_list.extend(glob.glob(file_glob))

        if not file_list:  # 文件为空
            print('No files found')
        else:
            for f in file_list:  # 扫描文件列表   这里f对应文件全路径
                filename = os.path.splitext(f.split("/")[-1])[0]  # 获取图片名字 hi
                # Data_zoo/MIT_SceneParsing/ADEChallengeData2016/annotations/training/*.png
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                if os.path.exists(annotation_file):  # 如果文件路径存在
                    #  image:图片全路径， annotation:标签全路径， filename:图片名字
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    # image_list{training:[{image:图片全路径， annotation:标签全路径， filename:图片名字}] [] []
                    #            validation:[{image:图片全路径， annotation:标签全路径， filename:图片名字}] [] []}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)
        # 对图片列表进行洗牌
        random.shuffle(image_list[directory])
        no_of_images = len(image_list[directory])  # 包含图片文件的个数
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list
