import datetime
import json
import re
import shutil
import time
from os import makedirs
from os.path import exists

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import constants


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def broadcast_iou(box_1, box_2):
    """ Code from: https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/dataset.py """
    # box_1: (..., (x1, y1, x2, y2))
    # box_2: (N, (x1, y1, x2, y2))

    # broadcast boxes
    box_1 = tf.expand_dims(box_1, -2)
    box_2 = tf.expand_dims(box_2, 0)
    # new_shape: (..., N, (x1, y1, x2, y2))
    new_shape = tf.broadcast_dynamic_shape(tf.shape(box_1), tf.shape(box_2))
    box_1 = tf.broadcast_to(box_1, new_shape)
    box_2 = tf.broadcast_to(box_2, new_shape)

    int_w = tf.maximum(tf.minimum(box_1[..., 2], box_2[..., 2]) -
                       tf.maximum(box_1[..., 0], box_2[..., 0]), 0)
    int_h = tf.maximum(tf.minimum(box_1[..., 3], box_2[..., 3]) -
                       tf.maximum(box_1[..., 1], box_2[..., 1]), 0)
    int_area = int_w * int_h
    box_1_area = (box_1[..., 2] - box_1[..., 0]) * \
                 (box_1[..., 3] - box_1[..., 1])
    box_2_area = (box_2[..., 2] - box_2[..., 0]) * \
                 (box_2[..., 3] - box_2[..., 1])
    return int_area / (box_1_area + box_2_area - int_area)


def create_recognition_ds_from_detection_ds(recognition_ds_name, detection_ds_name):
    """
    Create a dataset for image recognition from a dataset for image recognition.
    It will read and write the directories as in the constants files.
    It requires an annotations csv file with rows: [filename, xmin, ymin, xmax, ymax, label]
    The images will be saved in directories named wiht the label of those images.
    :param recognition_ds_name: Name of the recognition dataset.
    :param detection_ds_name: Name of the new dataset for image detection.
    """

    def create_ds(train_val_test, n_digits=None, extra_pos=0):
        """ n_digits and extra_pos are only for appending the validation set to the training one """
        annotations_df = pd.read_csv(constants.ANNOTATIONS_CSV_PATH.format(detection_ds_name, train_val_test))
        images_path = constants.DATASET_PATH.format(detection_ds_name) + train_val_test + '/'

        output = []
        if train_val_test == 'val':  # Train and validation folders end in the same train folder
            train_val_test = 'train'
            output = pd.read_csv(constants.DATASET_PATH.format(recognition_ds_name) + 'Train.csv').values.tolist()

        new_images_path = constants.DATASET_PATH.format(recognition_ds_name) + train_val_test + '/'
        if n_digits is None:
            n_digits = int(len(str(len(annotations_df))) * 1.5)

        for pos, row in tqdm(annotations_df.iterrows(), total=annotations_df.shape[0]):
            image_path = images_path + row['filename']
            img = cv2.imread(image_path)  # height, width, channels
            img_crop = img[row['ymin']:row['ymax'], row['xmin']:row['xmax'], :]
            label = str(row['label'])
            label_dir_path = new_images_path + (label if train_val_test == 'train' else '')
            dir_exists(label_dir_path)
            path = ('{:0' + str(n_digits) + 'd}.png').format(pos + extra_pos)
            img_path = label_dir_path + '/' + path
            cv2.imwrite(img_path, img_crop)
            output.append([row['xmax'] - row['xmin'], row['ymax'] - row['ymin'], label, path])

        output = pd.DataFrame(output, columns=['Width', 'Height', 'ClassId', 'Path'])
        output.to_csv(constants.DATASET_PATH.format(recognition_ds_name) + train_val_test.capitalize() + '.csv',
                      index=False)
        return n_digits, pos

    shutil.copy2(constants.LABEL_MAP_JSON_PATH.format(detection_ds_name),
                 constants.LABEL_MAP_JSON_PATH.format(recognition_ds_name))
    n_digits, pos = create_ds('train')
    create_ds('test')
    create_ds('val', n_digits, pos)


def dir_exists(directories_path):
    """ Check if the directories exist, if not create them and its intermediary folders """
    if type(directories_path) is not list:
        directories_path = [directories_path]

    for directory in directories_path:
        if not exists(directory):
            makedirs(directory)
            print(constants.C_WARNING, 'Created directory:', directory, constants.C_ENDC)


def display_time(seconds, granularity=4):
    result = []
    intervals = (
        ('weeks', 604800),  # 60 * 60 * 24 * 7
        ('days', 86400),  # 60 * 60 * 24
        ('hours', 3600),  # 60 * 60
        ('minutes', 60),
        ('seconds', 1),
    )

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    return ', '.join(result[:granularity])


def get_annotations_dict(dataset_name: str, train_val_test: str):
    """
    Get the dictionary with keys the filenames and values the list of bounding boxes of that image.
    If the annotations dictionary has not been created yet it will call the method to create it.
    """
    annotations_dict_path = constants.ANNOTATIONS_DICT_PATH.format(dataset_name, train_val_test)

    if exists(annotations_dict_path):
        with open(annotations_dict_path, "r") as ann_dict_file:
            annotations_dict = json.load(ann_dict_file)
        return annotations_dict
    return None


def get_labels_dict(dataset_name: str):
    with open(constants.LABEL_MAP_JSON_PATH.format(dataset_name)) as labels_dict_file:
        labels_dict = json.load(labels_dict_file)
    return labels_dict


def get_n_classes(json_path: str):
    """ Get the number of classes in the dataset that correspond to the json file """
    with open(json_path) as json_file:
        n_classes = len(json.load(json_file))
    return n_classes


def load_fake_dataset_detection(tiny=True):
    x_train = tf.image.decode_jpeg(
        open(constants.PROJECT_PATH + 'reports/test_image.jpg', 'rb').read(),
        channels=3)
    x_train = tf.expand_dims(x_train, axis=0)

    labels = [
                 [0.18494931, 0.03049111, 0.9435849, 0.96302897, 0],
                 [0.01586703, 0.35938117, 0.17582396, 0.6069674, 10],
                 [0.09158827, 0.48252046, 0.26967454, 0.6403017, 15]
             ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)
    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.batch(8)
    if tiny:
        dataset = dataset.map(lambda x, y: (transform_images(x, constants.YOLO_DEFAULT_IMAGE_RES),
                                            transform_targets(y, constants.TINY_YOLO_DEFAULT_ANCHORS,
                                                              constants.TINY_YOLO_DEFAULT_ANCHOR_MASKS)))
    else:
        dataset = dataset.map(lambda x, y: (transform_images(x, constants.YOLO_DEFAULT_IMAGE_RES),
                                            transform_targets(y, constants.YOLO_DEFAULT_ANCHORS,
                                                              constants.YOLO_DEFAULT_ANCHOR_MASKS)))

    return dataset


def load_fake_dataset_recognition(img_res):
    # Generate dummy data.
    data = np.random.random((5, 10, img_res, img_res, 3))
    labels = np.random.randint(10, size=(5, 10, 1))

    train_data = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(data), tf.convert_to_tensor(labels)))

    data = np.random.random((5, 10, img_res, img_res, 3))
    labels = np.random.randint(10, size=(5, 10, 1))

    val_data = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(data), tf.convert_to_tensor(labels)))
    return train_data, val_data


def model_directories(model_name: str, dataset_name: str):
    """
    Check if the directory for the model, the checkpoints and the logs exists and create them if not.
    The function makedirs creates all intermediary folders that do not exist
    :param model_name: Name of the model.
    :param dataset_name: Name of the dataset to use.
    :return: checkpoint path, logs path, figures path
    """

    date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '_' + dataset_name
    directories_path = [constants.MODELS_PROJECT_PATH.format(model_name, date) + path for path in constants.MODELS_DIRS]
    dir_exists(directories_path)

    return directories_path


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def save_history(FLAGS, model_name, dataset_name: str, history, start: float, det_rec: str):
    """
    Save the history of training in the reports folder.
    :param FLAGS: FLAGS used for training from absl.
    :param model_name: Name of the model used for training.
    :param dataset_name: Name of the dataset used for training.
    :param history: History from the fit function.
    :param start: Start of the training from time.time().
    :param det_rec: 'detection' or 'recognition'
    """
    train_time = display_time(int(time.time() - start), 2)
    val_losses = [model_name, dataset_name,
                  ('cosine' if FLAGS.use_cosine_lr else 'constant'), FLAGS.lr, train_time]
    cols = ['model', 'dataset', 'scheduler', 'initial_lr', 'train_time']
    if det_rec == 'detection':
        val_losses.append(FLAGS.trainable)
        cols.append('trainable')
    train_losses = val_losses.copy()

    for epoch in range(FLAGS.epochs):
        if epoch >= len(history.history['val_loss']):
            val_losses.append('-')
            train_losses.append('-')
        else:
            val_losses.append(round(history.history['val_loss'][epoch], 2))
            train_losses.append(round(history.history['loss'][epoch], 2))
        cols.append('epoch ' + str(epoch + 1))
    val_losses = pd.DataFrame([val_losses], columns=cols)
    train_losses = pd.DataFrame([train_losses], columns=cols)
    path = constants.PROJECT_PATH + '/reports/' + det_rec + '_{}_losses_epochs-' + str(FLAGS.epochs) + '.csv'
    if exists(path.format('val')):
        val_losses = pd.read_csv(path.format('val')).append(val_losses)
        train_losses = pd.read_csv(path.format('train')).append(train_losses)
    val_losses.to_csv(path.format('val'), index=False)
    train_losses.to_csv(path.format('train'), index=False)


def transform_images(img, size):
    """ Resize the image and normalize the values """
    return tf.image.resize(img, (size, size)) / 255


def transform_targets(y_train, anchors, anchor_masks):
    """ Code from: https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/dataset.py """
    y_outs = []
    grid_size = 13

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
                   tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    """ Code from: https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/dataset.py """
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1 / grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())
