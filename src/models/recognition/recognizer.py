import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2

import constants
import helpers


class Recognizer:
    def __init__(self, dataset_name: str, img_res: int, train: bool, model_name: str = "Recognizer"):
        self.model_name = model_name
        if dataset_name == '' or dataset_name == 'FAKE':
            self.dataset_name = 'FAKE'
            self.n_classes = 10
        else:
            self.dataset_name = dataset_name
            self.label_map_json_path = constants.LABEL_MAP_JSON_PATH.format(dataset_name)
            self.n_classes = helpers.get_n_classes(self.label_map_json_path)
            self.labels_map_dict = helpers.get_labels_dict(dataset_name)
        if train:
            self.checkpoints_path, self.logs_path, self.figs_path = helpers.model_directories(model_name,
                                                                                              self.dataset_name)
        self.img_res = img_res
        leaky_relu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
        self.train_model = tf.keras.models.Sequential([
            Conv2D(filters=32, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), activation=leaky_relu,
                   input_shape=(img_res, img_res, 3)),
            BatchNormalization(),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(filters=64, kernel_size=(3, 3), activation=leaky_relu, kernel_regularizer=l2(0.0005)),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(rate=0.25),
            Conv2D(filters=64, kernel_size=(3, 3), activation=leaky_relu, kernel_regularizer=l2(0.0005)),
            MaxPool2D(pool_size=(2, 2)),
            Dropout(rate=0.25),
            Conv2D(filters=128, kernel_size=(1, 1), activation=leaky_relu, kernel_regularizer=l2(0.0005)),
            Dropout(rate=0.5),
            Flatten(),
            Dense(self.n_classes, 'softmax')
        ])
        self.inference_model = self.train_model
        self.model_description = 'The model has {:,} parameters.'.format(self.train_model.count_params())
