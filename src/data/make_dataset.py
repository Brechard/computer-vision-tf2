import json
import os
from os.path import exists

import pandas as pd
import tensorflow as tf
from absl import app, flags
from tqdm import tqdm

import constants
import helpers
from data.external_to_raw import external_to_raw

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_name", '', "If the flag is empty then process all the dataset in constants.ALL_DATA_SETS")


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_list_feature(value):
    """Returns a float_list from a list of float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_annotations_dict(dataset_name: str, train_val_test: str):
    """ Creates a dictionary where the keys are the filenames and the values a list of the bounding boxes from csv
        for training, test or both if the flag is None """

    annotations_dict_path = constants.ANNOTATIONS_DICT_PATH.format(dataset_name, train_val_test)
    if exists(annotations_dict_path):
        return helpers.get_annotations_dict(dataset_name, train_val_test)
    print(constants.C_OKBLUE, 'data.make_dataset.create_annotations_dict. Started.', dataset_name,
          train_val_test, constants.C_ENDC)
    annotations_df = pd.read_csv(constants.ANNOTATIONS_CSV_PATH.format(dataset_name, train_val_test))
    annotations_dict = {}
    for index, bbox in tqdm(annotations_df.iterrows(), total=annotations_df.shape[0]):
        image_name = bbox.at['filename'].replace('.ppm', '')
        annotations_df.at[index, 'filename'] = image_name
        if image_name not in annotations_dict:
            annotations_dict[image_name] = []
        annotations_dict[image_name].append(
            [int(i) for i in annotations_df.iloc[index][['xmin', 'ymin', 'xmax', 'ymax', 'label']].values])

    with open(annotations_dict_path, 'w') as fp:
        json.dump(annotations_dict, fp, indent=4)
    print(constants.C_OKBLUE, 'data.make_dataset.create_annotations_dict. Finished.', dataset_name,
          train_val_test, constants.C_ENDC)
    return annotations_dict


def create_label_map_pbtxt(dataset_name: str):
    """ This kind of label map is used in the google models research repository """
    labels_dict = helpers.get_labels_dict(dataset_name)
    with open(constants.LABEL_MAP_PBTXT_PATH.format(dataset_name), 'a') as t:
        for key, value in labels_dict.items():
            id = str(int(key) + 1)  # pbtxt files are used in google models and they require id 0 to be background
            t.write("item {\n name: \"" + value + "\"\n id: " + id + "\n display_name: \"" + value + "\"\n}\n")


def create_tfrecords(dataset_name: str, train_val_test: str):
    """ TFRecords allow to create a uniform format that can then be used in several different models """
    tfrecords_path = constants.PROCESSED_PROJECT_FOLDER_PATH + constants.TFRECORDS_PATH

    images_path = constants.DATASET_PATH.format(dataset_name) + train_val_test + '/'
    annotations_dict = helpers.get_annotations_dict(dataset_name, train_val_test)
    if annotations_dict is None:
        annotations_dict = create_annotations_dict(dataset_name, train_val_test)

    print(constants.C_OKBLUE, 'data.make_dataset.create_tfrecords. Started for', dataset_name,
          train_val_test, constants.C_ENDC)
    images_list = os.listdir(images_path)
    index = 0
    n_images_shard = 800
    n_shards = int(len(images_list) / n_images_shard) + (1 if len(images_list) % 800 != 0 else 0)

    for shard in tqdm(range(n_shards)):
        tfrecords_shard_path = tfrecords_path.format(dataset_name, train_val_test,
                                                     '%.5d-of-%.5d' % (shard, n_shards - 1))
        end = index + n_images_shard if len(images_list) > (index + n_images_shard) else -1
        images_shard_list = images_list[index: end]
        with tf.io.TFRecordWriter(tfrecords_shard_path) as writer:
            for filename in images_shard_list:
                image_string = open(images_path + '/' + filename, 'rb').read()
                if filename in annotations_dict:
                    annotations_list = annotations_dict[filename]
                else:
                    annotations_list = [[0, 0, 0, 0, 0]]
                tf_example = image_example(image_string, annotations_list, filename)
                writer.write(tf_example.SerializeToString())
        index = end
    print(constants.C_OKGREEN, 'data.make_dataset.create_tfrecords. FINISHED for', dataset_name,
          train_val_test, constants.C_ENDC)


def image_example(image_encoded: str, annotations_list: list, filename: str) -> tf.train.Example:
    """
    Create an Example object with the features of the image. The bbox are normalized.
    :param image_encoded: Encoded image
    :param annotations_list: list with the bounding boxes characteristics
    :param filename: Filename as a string
    :return: tf.train.Example
    """
    height, width, _ = tf.io.decode_jpeg(image_encoded).shape
    x_min, y_min, x_max, y_max, labels = [], [], [], [], []
    if annotations_list is not None:
        for object_annotations in annotations_list:
            x_min.append(object_annotations[0] / width)
            y_min.append(object_annotations[1] / height)
            x_max.append(object_annotations[2] / width)
            y_max.append(object_annotations[3] / height)
            labels.append(object_annotations[4])

    feature = {
        'image/height': _int64_feature(height),
        'image/width': _int64_feature(width),
        'image/encoded': _bytes_feature(image_encoded),
        'image/filename': _bytes_feature(tf.compat.as_bytes(filename)),
        'image/object/bbox/xmin': _float_list_feature(x_min),
        'image/object/bbox/ymin': _float_list_feature(y_min),
        'image/object/bbox/xmax': _float_list_feature(x_max),
        'image/object/bbox/ymax': _float_list_feature(y_max),
        'image/object/class/label': _float_list_feature(labels),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_data(_argv):
    datasets = constants.ALL_DETECTION_DATA_SETS
    if FLAGS.dataset_name:
        datasets = [FLAGS.dataset_name]

    for dataset_name in datasets:
        external_to_raw[dataset_name]()
        create_label_map_pbtxt(dataset_name)
        for train_val_test in [constants.TRAIN, constants.VAL, constants.TEST]:
            try:
                create_annotations_dict(dataset_name, train_val_test)
                create_tfrecords(dataset_name, train_val_test)
            except FileNotFoundError as e:
                print(constants.C_FAIL, "Error when processing dataset", dataset_name, train_val_test, constants.C_ENDC)
                print(e)


if __name__ == '__main__':
    app.run(create_data)
