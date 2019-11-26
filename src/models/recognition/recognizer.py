import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, Dense, Flatten, Dropout
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
        self.train_model = tf.keras.models.Sequential([
            Conv2D(32, 5, 2, 'valid', kernel_regularizer=l2(0.0005), input_shape=(img_res, img_res, 3)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPool2D(2, 2, 'same'),
            Conv2D(64, 3, 1, kernel_regularizer=l2(0.0005)),
            LeakyReLU(alpha=0.1),
            MaxPool2D(2, 2, 'same'),
            Conv2D(128, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            LeakyReLU(alpha=0.1),
            Conv2D(256, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            Dropout(rate=0.5),
            Flatten(),
            Dense(self.n_classes, 'softmax')
        ])
        self.inference_model = self.train_model
        self.model_description = 'The model has {:,} parameters.'.format(self.train_model.count_params())
