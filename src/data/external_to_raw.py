"""
OBJECT RECOGNITION

This file contains the functions that will transform the files from external sources to raw data
that all have the same format so they can be preprocessed. Kind of like the preprocess of the preprocess.
Therefore each dataset will have their own functions tailored to their input formats.

Add the files to the folder: data/external/datasets/<DATASET_NAME> as they have been downloaded from
the original sources.
The images must be just under that folder with the structure:
    - train/
    - val/
    - test/ (optional)
If the downloaded files are not like this, the method should organize them so that they are. (Example in mapillary_ts)

The methods here should read the downloaded files from the original sources and create annotations files.
One for train, one for validation (val) and one for test if their labels are available.
The annotation file is just a csv file with the columns:
    filename, xmin, ymin, xmax, ymax, label
where each bounding box is a new row, the columns have self-explanatory names.

They should also create a dictionary mapping the label id with the label name when possible, saving it in:
    data/raw/<dataset_name>_label_map.json
it is recommended to use constants.LABEL_MAP_JSON_PATH.format(<dataset_name>).
If is is not possible to create it automatically (like with GTSD) then you should download it and add it
manually to the same folder.
"""
import imghdr
import json
import os
from os.path import exists

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import constants
import helpers

columns = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
val_proportion = 0.05


def convert_to_jpeg(val_index, test_index):
    """ Convert the PPM files to JPEG """

    def train_val_test(image_id):
        if int(image_id.split('.')[0]) < val_index:
            return 'train/' + image_id.replace('.ppm', '.jpeg')
        if int(image_id.split('.')[0]) < test_index:
            return 'val/' + image_id.replace('.ppm', '.jpeg')
        return 'test/' + image_id.replace('.ppm', '.jpeg')

    print(constants.C_OKBLUE, 'data.make_dataset.convert_to_jpeg. Started', constants.C_ENDC)
    jpeg_path = constants.DATASET_PATH.format(constants.GTSD)
    helpers.dir_exists([jpeg_path + extra for extra in ['train', 'val', 'test']])
    ppm_path = constants.DATASET_PATH.format(constants.GTSD)

    for index, name in enumerate(tqdm(os.listdir(ppm_path))):
        if '.ppm' not in name:
            continue
        new_image_path = jpeg_path + train_val_test(name)
        if exists(new_image_path):
            continue
        file_type = imghdr.what(ppm_path + name)
        if file_type != 'jpeg':
            image = cv2.imread(ppm_path + name)
            cv2.imwrite(new_image_path, image)

    print(constants.C_OKGREEN, 'data.make_dataset.convert_to_jpeg. Finished', constants.C_ENDC)


def gtsd():
    """
    The labels map files has to be in data/raw/GTSD_label_map.json since it cannot be created.
    (already added in the repository)
    """
    label_map_path = constants.LABEL_MAP_JSON_PATH.format(constants.GTSD)
    if not exists(label_map_path):
        raise FileNotFoundError('Label map for the dataset ' + constants.GTSD + ' not found in ' + label_map_path)
    test_index = 700
    val_index = int(test_index * (1 - val_proportion))
    convert_to_jpeg(val_index, test_index)
    labels_path = constants.DATASET_PATH.format(constants.GTSD) + 'gt.txt'
    labels = pd.read_csv(labels_path, sep=';', names=columns)
    labels['filename'] = labels['filename'].apply(lambda x: x.replace('ppm', 'jpeg'))
    file_names = labels['filename'].apply(lambda x: int(x.replace('.jpeg', '')))
    train_val_df = labels[file_names < test_index]
    file_names_train_val = train_val_df['filename'].apply(lambda x: int(x.replace('.jpeg', '')))

    train_indexes = file_names_train_val < val_index
    val_indexes = file_names_train_val >= val_index

    train_df = train_val_df[train_indexes]
    val_df = train_val_df[val_indexes]
    test_df = labels[file_names >= test_index]

    train_df.to_csv(constants.ANNOTATIONS_CSV_PATH.format(constants.GTSD, constants.TRAIN), index=False)
    val_df.to_csv(constants.ANNOTATIONS_CSV_PATH.format(constants.GTSD, constants.VAL), index=False)
    test_df.to_csv(constants.ANNOTATIONS_CSV_PATH.format(constants.GTSD, constants.TEST), index=False)


def bdd100k():
    """
    Creates the label map automatically. Requires the downloaded labels JSON files in:
        data/external/BDD100K_labels_train.json
        data/external/BDD100K_labels_val.json
    In this case the labels are as text so we have to give them an id. Therefore and inverse map
    dictionary is very helpful (from label text to id).
    """

    def helper(f, save_path):
        annotations = []
        for frame in tqdm(f):
            for label in frame['labels']:
                if 'box2d' not in label:
                    continue
                xy = label['box2d']
                if xy['x1'] >= xy['x2'] or xy['y1'] >= xy['y2']:
                    continue
                if label['category'] not in labels_map_dict_inverse.keys():
                    labels_map_dict[len(labels_map_dict_inverse.keys())] = label['category']
                    labels_map_dict_inverse[label['category']] = len(labels_map_dict_inverse.keys())

                label = labels_map_dict_inverse[label['category']]
                annotation = [frame['name'], int(xy['x1']), int(xy['y1']), int(xy['x2']), int(xy['y2']), label]
                annotations.append(annotation)
        pd.DataFrame(annotations, columns=columns).to_csv(save_path, index=False)

    labels_map_dict = {}
    labels_map_dict_inverse = {}
    frames = json.load(open(constants.EXTERNAL_ANNOTATIONS_TRAIN_PATH.format(constants.BDD100K, 'json'), 'r'))
    helper(frames, constants.ANNOTATIONS_CSV_PATH.format(constants.BDD100K, constants.TRAIN))

    frames = json.load(open(constants.EXTERNAL_ANNOTATIONS_VAL_PATH.format(constants.BDD100K, 'json'), 'r'))
    helper(frames, constants.ANNOTATIONS_CSV_PATH.format(constants.BDD100K, constants.VAL))

    with open(constants.LABEL_MAP_JSON_PATH.format(constants.BDD100K), 'w') as fp:
        json.dump(labels_map_dict, fp, indent=4)


def mapillary_ts():
    """
    This is a private DS, if you want to use it, you have to require access as a researcher or purchase it.
    Just add everything that was downloaded from the link to the folder that you define in constants.MAPILLARY_TS_PATH
    """

    def helper(file_list, save_path):
        annotations = []
        for file in tqdm(file_list):
            file_annotations_dict = json.load(open(ds_path + 'annotations/' + file + '.json', 'r'))
            for annotation in file_annotations_dict['objects']:
                if annotation['label'] not in labels_map_dict_inverse.keys():
                    labels_map_dict[len(labels_map_dict_inverse.keys())] = annotation['label']
                    labels_map_dict_inverse[annotation['label']] = len(labels_map_dict_inverse.keys())

                label = labels_map_dict_inverse[annotation['label']]
                bbox = annotation['bbox']
                annotations.append(
                    [file + 'jpg', int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax']), label])
        pd.DataFrame(annotations, columns=columns).to_csv(save_path, index=False)

    ds_path = constants.DATASET_PATH.format(constants.MAPILLARY_TS)
    labels_map_dict = {}
    labels_map_dict_inverse = {}
    train_files = open(ds_path + "splits/train.txt", "r").read().splitlines()
    val_files = open(ds_path + "splits/val.txt", "r").read().splitlines()

    helper(train_files, constants.ANNOTATIONS_CSV_PATH.format(constants.MAPILLARY_TS, constants.TRAIN))
    helper(val_files, constants.ANNOTATIONS_CSV_PATH.format(constants.MAPILLARY_TS, constants.VAL))

    with open(constants.LABEL_MAP_JSON_PATH.format(constants.MAPILLARY_TS), 'w') as fp:
        json.dump(labels_map_dict, fp, indent=4)


def coco():
    """
    Labels map will be created automatically but needs the original json file (already included in this repository).
    """

    def helper(path, label_map_correct_map, save_path):
        annotations_dict = json.load(
            open(constants.DATASET_PATH.format(constants.COCO) + 'annotations_trainval2017/annotations/' + path, "r"))
        image_id_dict = {}
        annotations = []

        for image in annotations_dict['images']:
            image_id_dict[image['id']] = image['file_name']
        for annotation in tqdm(annotations_dict['annotations']):
            bbox = list(np.array(annotation['bbox'], dtype=int))
            label_id = label_map_correct_map[annotation['category_id']]
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
            annotations.append(
                [image_id_dict[annotation['image_id']], x_min, y_min, x_max, y_max, label_id])
        pd.DataFrame(annotations, columns=columns).to_csv(save_path, index=False)

    labels_map_correct_map = {}  # COCO has 90 ids in the labels map but in the dataset there are only 80!
    if os.path.exists(constants.LABEL_MAP_JSON_PATH.format(constants.COCO)) and os.path.exists(
            constants.LABEL_MAP_JSON_PATH.format(constants.COCO + '_mapping')):
        labels_map_correct_map = json.load(open(constants.LABEL_MAP_JSON_PATH.format(constants.COCO + '_mapping'), 'r'))
    else:
        labels_map_dict = {}
        labels_map_dict_pre = json.load(open(constants.EXTERNAL_PATH + '/coco.json', 'r'))
        for obj in labels_map_dict_pre:
            correct_id = len(labels_map_dict)
            labels_map_dict[correct_id] = obj['name']
            labels_map_correct_map[obj['id']] = correct_id
        with open(constants.LABEL_MAP_JSON_PATH.format(constants.COCO), 'w') as fp:
            json.dump(labels_map_dict, fp, indent=4)

    helper("instances_train2017.json", labels_map_correct_map,
           constants.ANNOTATIONS_CSV_PATH.format(constants.COCO, constants.TRAIN))
    helper("instances_val2017.json", labels_map_correct_map,
           constants.ANNOTATIONS_CSV_PATH.format(constants.COCO, constants.VAL))


external_to_raw = {
    constants.GTSD: gtsd,
    constants.MAPILLARY_TS: mapillary_ts,
    constants.BDD100K: bdd100k,
    constants.COCO: coco
}
