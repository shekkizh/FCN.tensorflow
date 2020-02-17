__author__ = 'charlie'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
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


# def read_dataset(data_dir):
#     pickle_filename = "MITSceneParsing.pickle"
#     pickle_filepath = os.path.join(data_dir, pickle_filename)
#     if not os.path.exists(pickle_filepath):
#         utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
#         SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
#         result = create_image_lists(os.path.join(data_dir, SceneParsing_folder))
#         print ("Pickling ...")
#         with open(pickle_filepath, 'wb') as f:
#             pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
#     else:
#         print ("Found pickle file!")

#     with open(pickle_filepath, 'rb') as f:
#         result = pickle.load(f)
#         training_records = result['training']
#         validation_records = result['validation']
#         del result

#     return training_records, validation_records


# def create_image_lists(image_dir):
#     if not gfile.Exists(image_dir):
#         print("Image directory '" + image_dir + "' not found.")
#         return None
#     directories = ['training', 'validation']
#     image_list = {}

#     for directory in directories:
#         file_list = []
#         image_list[directory] = []
#         file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
#         file_list.extend(glob.glob(file_glob))

#         if not file_list:
#             print('No files found')
#         else:
#             for f in file_list:
#                 filename = os.path.splitext(f.split("\\")[-1])[0]
#                 annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
#                 if os.path.exists(annotation_file):
#                     record = {'image': f, 'annotation': annotation_file, 'filename': filename}
#                     image_list[directory].append(record)
#                 else:
#                     print("Annotation file not found for %s - Skipping" % filename)

#         random.shuffle(image_list[directory])
#         no_of_images = len(image_list[directory])
#         print ('No. of %s files: %d' % (directory, no_of_images))

#     return image_list
