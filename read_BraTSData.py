__author__ = 'charlie'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
# DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
TRAIN_SPLIT = 160

def read_dataset(data_dir):
    pickle_filename = 'BraTS_pickle.txt'
    pickle_path = os.path.join(data_dir, pickle_filename)

    if not os.path.exists(pickle_path):
        records = create_records(data_dir)
        train_records, val_records = split_train_val(records, TRAIN_SPLIT)
        record_dic = {
            "training": train_records,
            "validation": val_records
        }
        with open(pickle_path, 'wb') as f:
            pickle.dump(record_dic, f)
    else:
        with open(pickle_path, 'rb') as f:
            record_dic = pickle.load(f)
            train_records = record_dic["training"]
            val_records = record_dic["validation"]
            
    return train_records, val_records


def create_records(data_dir):
    dir_list = os.listdir(data_dir)
    
    records = []

    for dir in dir_list:
        path = os.path.join(data_dir, dir)
        file_list = glob.glob(path + "\\*.gz")
        seg_list = [file for file in file_list if file.endswith("seg.nii.gz")]
        file_list.remove(seg_list[0])
        record = {
            "input": file_list,
            "mask": seg_list,
            "filename": dir
        }
        records.insert(0, record)
        
    return records


def split_train_val(records, train_split):
    random.shuffle(records)
    train_records = records[:train_split]
    val_records = records[train_split:]
    
    return train_records, val_records
