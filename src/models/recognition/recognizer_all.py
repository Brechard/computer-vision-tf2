import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2

import constants
import helpers

versions = ['mini', 'mini_2', 'v0', 'v0.1', 'v0.2', 'v0.3', 'v0.4', 'v1', 'v2', 'v3', 'mod', 'mod_bn',
            'mod_bn_reg', 'kaggle']


class Recognizer:
    def __init__(self, dataset_name: str, img_res: int, train: bool, model_version: str,
                 train_cluster: bool = False):
        self.model_name = "Recognizer_" + model_version
        if dataset_name == '':
            self.dataset_name = 'FAKE'
            self.n_classes = 10
        else:
            self.dataset_name = dataset_name
            self.label_map_json_path = constants.LABEL_MAP_JSON_PATH.format(dataset_name)
            self.n_classes = helpers.get_n_classes(self.label_map_json_path)
            self.labels_map_dict = helpers.get_labels_dict(dataset_name)
        if train:
            self.checkpoints_path, self.logs_path, self.figs_path = helpers.model_directories(constants.RECOGNIZER,
                                                                                              self.dataset_name,
                                                                                              train_cluster)
        self.model_description = ''
        if model_version == 'mini':
            self.train_model = self.model_mini(img_res)
        elif model_version == 'mini_2':
            self.train_model = self.model_mini_2(img_res)
        elif model_version == 'kaggle':
            self.train_model = self.kaggle_model(img_res)
        elif model_version == 'mod':
            self.train_model = self.mod(img_res)
        elif model_version == 'mod_bn':
            self.train_model = self.mod_bn(img_res)
        elif model_version == 'mod_bn_reg':
            self.train_model = self.mod_bn_reg(img_res)
        elif model_version == 'v0':
            self.train_model = self.model_v0(img_res)
        elif model_version == 'v0.1':
            self.train_model = self.model_v0_1(img_res)
        elif model_version == 'v0.2':
            self.train_model = self.model_v0_2(img_res)
        elif model_version == 'v0.3':
            self.train_model = self.model_v0_3(img_res)
        elif model_version == 'v0.4':
            self.train_model = self.model_v0_4(img_res)
        elif model_version == 'v1':
            self.train_model = self.model_v1(img_res)
        elif model_version == 'v2':
            self.train_model = self.model_v2(img_res)
        elif model_version == 'v3':
            self.train_model = self.model_v3(img_res)

        self.img_res = img_res
        self.model_description += 'The model has {:,} parameters.'.format(self.train_model.count_params())
        self.inference_model = self.train_model

    def model_v0(self, input_shape):
        self.model_description = 'Version 0 of the classifier. Using BatchNorm.'
        return tf.keras.models.Sequential([
            Conv2D(32, 5, 2, 'valid', kernel_regularizer=l2(0.0005), input_shape=(input_shape, input_shape, 3)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPool2D(2, 2, 'same'),
            Conv2D(64, 3, 1, kernel_regularizer=l2(0.0005)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPool2D(2, 2, 'same'),
            Conv2D(128, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Conv2D(256, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            Flatten(),
            Dense(self.n_classes, 'softmax')
        ])

    def model_v0_1(self, input_shape):
        self.model_description = 'Version 0.1 of the classifier. Using BatchNorm. No activations.'
        return tf.keras.models.Sequential([
            Conv2D(32, 5, 2, 'valid', kernel_regularizer=l2(0.0005), input_shape=(input_shape, input_shape, 3)),
            BatchNormalization(),
            MaxPool2D(2, 2, 'same'),
            Conv2D(64, 3, 1, kernel_regularizer=l2(0.0005)),
            BatchNormalization(),
            MaxPool2D(2, 2, 'same'),
            Conv2D(128, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            BatchNormalization(),
            Conv2D(256, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            Flatten(),
            Dense(self.n_classes, 'softmax')
        ])

    def model_v0_2(self, input_shape):
        self.model_description = 'Version 0.2 of the classifier. Using 2 BatchNorm.'
        return tf.keras.models.Sequential([
            Conv2D(32, 5, 2, 'valid', kernel_regularizer=l2(0.0005), input_shape=(input_shape, input_shape, 3)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPool2D(2, 2, 'same'),
            Conv2D(64, 3, 1, kernel_regularizer=l2(0.0005)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPool2D(2, 2, 'same'),
            Conv2D(128, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            LeakyReLU(alpha=0.1),
            Conv2D(256, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            Flatten(),
            Dense(self.n_classes, 'softmax')
        ])

    def model_v0_3(self, input_shape):
        self.model_description = 'Version 0.3 of the classifier. Using only 1 BatchNorm.'
        return tf.keras.models.Sequential([
            Conv2D(32, 5, 2, 'valid', kernel_regularizer=l2(0.0005), input_shape=(input_shape, input_shape, 3)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPool2D(2, 2, 'same'),
            Conv2D(64, 3, 1, kernel_regularizer=l2(0.0005)),
            LeakyReLU(alpha=0.1),
            MaxPool2D(2, 2, 'same'),
            Conv2D(128, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            LeakyReLU(alpha=0.1),
            Conv2D(256, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            Flatten(),
            Dense(self.n_classes, 'softmax')
        ])

    def model_v0_4(self, input_shape):
        self.model_description = 'Version 0.4 of the classifier. Using only 1 BatchNorm. End dropout.'
        return tf.keras.models.Sequential([
            Conv2D(32, 5, 2, 'valid', kernel_regularizer=l2(0.0005), input_shape=(input_shape, input_shape, 3)),
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

    def model_mini(self, input_shape):
        self.model_description = 'Version MINI CLASSIFIER!!!.'
        return tf.keras.models.Sequential([
            Conv2D(32, 5, 2, 'valid', kernel_regularizer=l2(0.0005), input_shape=(input_shape, input_shape, 3)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(64, 3, 2, 'valid', kernel_regularizer=l2(0.0005)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPool2D(pool_size=(2, 2)),
            Flatten(),
            Dense(self.n_classes, 'softmax')
        ])

    def model_mini_2(self, input_shape):
        self.model_description = 'Version 2 of MINI CLASSIFIER!!!.'
        return tf.keras.models.Sequential([
            Conv2D(32, 5, 2, 'valid', kernel_regularizer=l2(0.0005), input_shape=(input_shape, input_shape, 3)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(64, 3, 2, 'valid', kernel_regularizer=l2(0.0005)),
            LeakyReLU(alpha=0.1),
            MaxPool2D(pool_size=(2, 2)),
            Conv2D(64, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            LeakyReLU(alpha=0.1),
            Dropout(0.5),
            Flatten(),
            Dense(self.n_classes, 'softmax')
        ])

    def model_v1(self, input_shape):
        self.model_description = 'Version 1 of the classifier. Using BatchNorm and dropout.'
        return tf.keras.models.Sequential([
            Conv2D(32, 5, 2, 'valid', kernel_regularizer=l2(0.0005), input_shape=(input_shape, input_shape, 3)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(0.25),
            MaxPool2D(2, 2, 'same'),
            Conv2D(64, 3, 1, kernel_regularizer=l2(0.0005)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(0.25),
            MaxPool2D(2, 2, 'same'),
            Conv2D(128, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            BatchNormalization(),
            LeakyReLU(alpha=0.1),
            Dropout(0.25),
            Conv2D(256, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            Dropout(0.25),
            Flatten(),
            Dense(self.n_classes, 'softmax')
        ])

    def model_v2(self, input_shape):
        self.model_description = 'Version 2 of the classifier. No BatchNorm but using dropout.'
        return tf.keras.models.Sequential([
            Conv2D(32, 5, 2, 'valid', kernel_regularizer=l2(0.0005), input_shape=(input_shape, input_shape, 3)),
            Dropout(0.25),
            MaxPool2D(2, 2, 'same'),
            Conv2D(64, 3, 1, kernel_regularizer=l2(0.0005)),
            Dropout(0.25),
            MaxPool2D(2, 2, 'same'),
            Conv2D(128, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            Dropout(0.25),
            Conv2D(256, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            Dropout(0.25),
            Flatten(),
            Dense(self.n_classes, 'softmax')
        ])

    def model_v3(self, input_shape):
        self.model_description = 'Version 3 of the classifier. No BatchNorm, no dropout. No activation'
        return tf.keras.models.Sequential([
            Conv2D(32, 5, 2, 'valid', kernel_regularizer=l2(0.0005), input_shape=(input_shape, input_shape, 3)),
            MaxPool2D(2, 2, 'same'),
            Conv2D(64, 3, 1, kernel_regularizer=l2(0.0005)),
            MaxPool2D(2, 2, 'same'),
            Conv2D(128, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            Conv2D(256, 1, 1, 'valid', kernel_regularizer=l2(0.0005)),
            Flatten(),
            Dense(self.n_classes, 'softmax')
        ])

    def mod(self, input_shape):
        self.model_description = 'Modified model to use conv instead of Dense layer.'
        model = tf.keras.models.Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(input_shape, input_shape, 3)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=128, kernel_size=(1, 1), activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Flatten())
        model.add(Dense(self.n_classes, activation='softmax'))
        return model

    def mod_bn(self, input_shape):
        self.model_description = 'Modified model with 1 BatchNormalization.'
        model = tf.keras.models.Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(input_shape, input_shape, 3)))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=128, kernel_size=(1, 1), activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Flatten())
        model.add(Dense(self.n_classes, activation='softmax'))
        return model

    def mod_bn_reg(self, input_shape):
        self.model_description = 'Modified model with 1 BatchNormalization and kernel regularizers.'
        model = tf.keras.models.Sequential()
        leaky_relu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
        model.add(Conv2D(filters=32, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), activation=leaky_relu,
                         input_shape=(input_shape, input_shape, 3)))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation=leaky_relu, kernel_regularizer=l2(0.0005)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation=leaky_relu, kernel_regularizer=l2(0.0005)))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=128, kernel_size=(1, 1), activation=leaky_relu, kernel_regularizer=l2(0.0005)))
        model.add(Dropout(rate=0.5))
        model.add(Flatten())
        model.add(Dense(self.n_classes, activation='softmax'))
        return model

    def kaggle_model(self, input_shape):
        self.model_description = 'Version found in kaggle.'
        model = tf.keras.models.Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(input_shape, input_shape, 3)))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(self.n_classes, activation='softmax'))
        return model
