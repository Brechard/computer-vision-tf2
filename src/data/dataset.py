import tensorflow as tf

import constants
import data.preprocessing.preprocessing as preprocessing
import helpers


class Dataset:
    def __init__(self, dataset_name: str, tiny: bool = True):
        if dataset_name == '':
            self.dataset_name = 'FAKE'
            self.n_classes = 80
            self.tiny = tiny
            print(constants.C_OKBLUE, "FAKE dataset created", constants.C_ENDC)
            return
        self.dataset_name = dataset_name
        self.label_map_json_path = constants.LABEL_MAP_JSON_PATH.format(dataset_name)
        self.tf_paths = constants.PROCESSED_PROJECT_FOLDER_PATH + constants.TFRECORDS_PATH
        self.n_classes = helpers.get_n_classes(self.label_map_json_path)
        self.labels_map_dict = helpers.get_labels_dict(dataset_name)
        self.train_data, self.validation_data, self.test_data = None, None, None
        self.image_res, self.loading_train = None, None  # To be defined when called load_datasets
        print(constants.C_OKBLUE, "Dataset", dataset_name, "created", constants.C_ENDC)

    def load_datasets(self, image_res: int, anchors: list, masks: list, batch_size: int) -> None:
        def get_dataset_from_pattern(tfrecords_pattern_path, train=False):
            """ Get the data from tfrecords """
            files = tf.random.shuffle(tf.io.matching_files(tfrecords_pattern_path))
            shards = tf.data.Dataset.from_tensor_slices(files)
            if train:
                shards = shards.repeat(10)
            dataset = shards.interleave(tf.data.TFRecordDataset)
            dataset = dataset.shuffle(buffer_size=800)
            # for i in dataset:
            #     self.parse_tfrecord(i)
            #     pib((x_train + 1) / 2, y_train, dataset_name="COCO")
            dataset = dataset.map(map_func=self.parse_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            # for x, y in dataset.take(5):
            #     a = 1
            #     pib((x + 1) / 2, y, dataset_name="COCO")
            #
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(map_func=lambda x, y: (x,
                                                         helpers.transform_targets(y, anchors,
                                                                                   masks)),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
            if train:
                dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return dataset

        if self.dataset_name == 'FAKE':
            self.train_data = helpers.load_fake_dataset_detection(self.tiny)
            self.validation_data = helpers.load_fake_dataset_detection(self.tiny)
            print(constants.C_OKBLUE, "FAKE Dataset loaded", constants.C_ENDC)
            return
        self.image_res = image_res
        self.loading_train = True
        self.train_data = get_dataset_from_pattern(self.tf_paths.format(self.dataset_name, 'train', '*-of-*'), True)
        self.loading_train = False
        self.validation_data = get_dataset_from_pattern(self.tf_paths.format(self.dataset_name, 'val', '*-of-*'))
        self.test_data = get_dataset_from_pattern(self.tf_paths.format(self.dataset_name, 'test', '*-of-*'))

        print(constants.C_OKBLUE, "Dataset", self.dataset_name, "loaded", constants.C_ENDC)

    def parse_tfrecord(self, tfrecord, preprocess=True):
        x = tf.io.parse_single_example(tfrecord, constants.IMAGE_FEATURE_MAP)
        x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
        y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                            tf.sparse.to_dense(x['image/object/bbox/ymin']),
                            tf.sparse.to_dense(x['image/object/bbox/xmax']),
                            tf.sparse.to_dense(x['image/object/bbox/ymax']),
                            tf.sparse.to_dense(x['image/object/class/label'])], axis=1)
        if preprocess:
            x_train, y_train = preprocessing.preprocess_image(image=x_train, output_height=self.image_res,
                                                              output_width=self.image_res,
                                                              is_training=self.loading_train,
                                                              bboxes=tf.convert_to_tensor(y_train))

            paddings = [[0, 100 - tf.shape(y_train)[0]], [0, 0]]
            y_train = tf.pad(y_train, paddings)
        return x_train, y_train
