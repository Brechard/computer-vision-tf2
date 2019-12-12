from os.path import abspath, dirname

import numpy as np
import tensorflow as tf

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

""" 

Available DataSets 

"""
GTSR = 'GTSR'
GTSD = 'GTSD'
BDD100K = 'BDD100K'
MAPILLARY_TS = 'MAPILLARY_TS'
COCO = 'COCO'

ALL_DETECTION_DATA_SETS = [GTSD,
                           BDD100K,
                           MAPILLARY_TS,
                           COCO]

"""

PATH

"""

# Inside the project
PROJECT_PATH = abspath(dirname(__file__)).split('/src')[0] + '/'

""" DATA paths """
#   External path
#       Here we have to copy the labels files we get from the dataset that we then convert to a json into the
#       the raw folder with the same format in every dataset. Meaning:
#           - A json label map, key the id (integer) of the label and value the label value (string)
#           - A csv file with: filename, xmin, ymin, xmax, ymax, label
#       TRAIN and VALIDATION labels have to be grouped since we will use k-fold cross validation

EXTERNAL_PATH = PROJECT_PATH + 'data/external/'
DATASET_PATH = EXTERNAL_PATH + 'datasets/{}/'
EXTERNAL_ANNOTATIONS_PATH = EXTERNAL_PATH + '{}_labels.{}'
EXTERNAL_ANNOTATIONS_TRAIN_PATH = EXTERNAL_PATH + '{}_labels_train.{}'
EXTERNAL_ANNOTATIONS_TEST_PATH = EXTERNAL_PATH + '{}_labels_test.{}'
EXTERNAL_ANNOTATIONS_VAL_PATH = EXTERNAL_PATH + '{}_labels_val.{}'

#   Raw paths
RAW_PATH = PROJECT_PATH + 'data/raw/'
# key (integer): the id  of the label
# value (string): the label value
LABEL_MAP_JSON_PATH = RAW_PATH + '{}_label_map.json'

# filename, xmin, ymin, xmax, ymax, label
ANNOTATIONS_CSV_PATH = RAW_PATH + '{}_annotations_{}.csv'
#                                  DATASET_annotations_TRAIN/VAL/TEST.csv
#   Interim paths
INTERIM_FOLDER_PATH = PROJECT_PATH + 'data/interim/'
# key (string): filename
# value (list): list of bounding boxes that correspond to the filename each with format:
#                   [x_min, y_min, x_max, y_max, label]
ANNOTATIONS_DICT_PATH = INTERIM_FOLDER_PATH + 'annotations_{}_{}_dict.json'
#                                              annotations_DATASET_TRAIN/VAL/TEST_dict.p

#   Processed paths
PROCESSED_PROJECT_FOLDER_PATH = PROJECT_PATH + 'data/processed/'

TFRECORDS_PATH = '{}_{}_{}.records'
#                 DATASET_TRAIN/VAL/TEST_SHARD-NSHARDS.records

""" Models """
MODELS_PROJECT_PATH = PROJECT_PATH + 'models/{}/{}/'
CHECKPOINTS = 'checkpoints/'
LOGS = 'logs/'
FIGURES = 'figures/'
MODELS_DIRS = [CHECKPOINTS, LOGS, FIGURES]

YOLOv3 = 'YOLOv3'
Tiny_YOLOv3 = 'Tiny_YOLOv3'

RECOGNIZER = "Recognizer"

"""

Model

"""
IMAGE_FEATURE_MAP = {
    'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/label': tf.io.VarLenFeature(tf.float32),
}
TRAINABLE_ALL = "all"
TRAINABLE_FEATURES = "features"
TRAINABLE_LAST_BLOCK = "last_block"
TRAINABLE_LAST_CONV = "last_conv"

# YOLO
YOLO_DEFAULT_IMAGE_RES = 416
YOLO_DEFAULT_ANCHORS = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),
                                 (59, 119), (116, 90), (156, 198), (373, 326)],
                                np.float32) / YOLO_DEFAULT_IMAGE_RES
YOLO_DEFAULT_ANCHOR_MASKS = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

TINY_YOLO_DEFAULT_ANCHORS = np.array([(10, 14), (23, 27), (37, 58),
                                      (81, 82), (135, 169), (344, 319)],
                                     np.float32) / YOLO_DEFAULT_IMAGE_RES
TINY_YOLO_DEFAULT_ANCHOR_MASKS = np.array([[3, 4, 5], [0, 1, 2]])

""" 

Visualization 

"""

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
          (255, 128, 0), (128, 0, 255), (255, 0, 128),
          (255, 255, 0), (255, 0, 255), (0, 255, 255),
          (205, 128, 77), (128, 0, 128), (0, 128, 128),
          (255, 255, 255), (128, 128, 128), (0, 0, 0)]

# Colors for printing in command line
C_HEADER = '\033[95m'
C_OKBLUE = '\033[94m'
C_OKGREEN = '\033[92m'
C_WARNING = '\033[33m'
C_FAIL = '\033[91m'
C_ENDC = '\033[0m'
C_BOLD = '\033[1m'
C_UNDERLINE = '\033[4m'
