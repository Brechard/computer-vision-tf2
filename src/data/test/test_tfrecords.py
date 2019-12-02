import os
import unittest

import cv2
import numpy as np
import tensorflow as tf

import constants
from data import make_dataset
from data.dataset import Dataset


class TFRecordTest(unittest.TestCase):
    def test_create_tfrecords(self):
        dataset_name = constants.MAPILLARY_TS
        dataset = Dataset(dataset_name)
        images_path = constants.DATASET_PATH.format(dataset_name) + '/train/'
        annotations_dict = make_dataset.create_annotations_dict(dataset_name, 'train')
        no_annotations, total_analyzed = 0, 0
        for i, filename in enumerate(os.listdir(images_path)):
            if np.random.random() >= 0.01:
                continue
            total_analyzed += 1
            image_path = images_path + filename
            image_string = open(images_path + '/' + filename, 'rb').read()
            annotations = []
            if filename in annotations_dict:
                height, width, _ = cv2.imread(image_path).shape
                annotations_list = annotations_dict[filename]
                for annotation in annotations_list:
                    annotations.append([annotation[0] / width, annotation[1] / height,
                                        annotation[2] / width, annotation[3] / height,
                                        annotation[4]])
            else:
                annotations = annotations_list = [[0.0, 0.0, 0.0, 0.0, 0.0]]
                no_annotations += 1
            annotations = np.array(annotations, dtype=np.float32)

            tf_example = make_dataset.image_example(image_string, annotations_list, filename)

            file_path = 'data.tfrecords'
            with tf.io.TFRecordWriter(file_path) as writer:
                writer.write(tf_example.SerializeToString())

            tf_records = tf.data.TFRecordDataset([file_path])
            for record in tf_records:
                image, labels = dataset.parse_tfrecord(record, False)
                labels = labels.numpy()
                self.assertEqual(np.sum(annotations != labels), 0)
        self.assertNotEqual(total_analyzed, no_annotations)


if __name__ == '__main__':
    unittest.main()
