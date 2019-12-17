import os

import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

from helpers import natural_keys


def get_items_list_by_index(items_list, indexes):
    return [items_list[idx] for idx in indexes]


class Generator:
    def __init__(self, directory, batch_size, image_dimensions=(50, 50), validation_split=0.1):
        """
        Similar to keras.preprocessing.image.ImageDataGenerator it will load the images from the
        indicated directory, where images should be stored in folders with name the label of the
        class. It divides them in train and validation splits after shuffle and creates image
        generators for each. Train split has data augmentation techniques.

        :param directory: Path to the directory with the folders that contain the images.
        :param batch_size: batch size.
        :param image_dimensions: Dimension of the images to use.
        :param validation_split: Percentage that corresponds to the validation split.
        """
        self.dim = image_dimensions
        self.batch_size = batch_size
        self.classes = sorted(os.listdir(directory), key=natural_keys)
        self.n_classes = len(self.classes)
        self.class_indices = {}
        for index, class_label in enumerate(self.classes):
            self.class_indices[class_label] = index

        self.labels = []
        self.images_paths = []
        for class_id in self.classes:
            class_dir = os.path.join(directory, class_id)
            self.images_paths.extend([os.path.join(class_dir, image_path) for image_path in os.listdir(class_dir)])
            self.labels.extend([class_id] * len(os.listdir(class_dir)))

        indexes = np.arange(len(self.images_paths))
        np.random.shuffle(indexes)
        self.train_indexes = indexes[:int((1 - validation_split) * len(indexes))]
        self.val_indexes = indexes[int((1 - validation_split) * len(indexes)):]
        self.train_generator = DataGenerator(get_items_list_by_index(self.images_paths, self.train_indexes),
                                             get_items_list_by_index(self.labels, self.train_indexes),
                                             True, self.n_classes, batch_size, image_dimensions)
        self.val_generator = DataGenerator(get_items_list_by_index(self.images_paths, self.val_indexes),
                                           get_items_list_by_index(self.labels, self.val_indexes),
                                           False, self.n_classes, batch_size, image_dimensions)


class DataGenerator(Sequence):
    def __init__(self, images_paths, labels, augment, n_classes, batch_size=256, image_dimensions=(50, 50)):
        self.dim = image_dimensions  # image dimensions
        self.batch_size = batch_size  # batch size
        self.n_classes = n_classes
        self.augment = augment
        self.labels = labels
        self.images_paths = images_paths
        self.indexes = np.arange(len(self.images_paths))
        np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """ Generate one batch of data """
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]

        # select data and load images
        labels = np.array([self.labels[k] for k in indexes])
        images = np.array([self.get_image(self.images_paths[k]) for k in indexes])

        if self.augment:
            # preprocess and augment data
            images = seq.augment_images(images) / 255

        return images, to_categorical(labels, self.n_classes)

    def __len__(self):
        """ Denotes the number of batches per epoch """
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        """ Updates indexes after each epoch """
        self.indexes = np.arange(len(self.images_paths))
        np.random.shuffle(self.indexes)

    def get_image(self, img_path):
        return cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), self.dim)


def sometimes(aug):
    return iaa.Sometimes(0.5, aug)


seq = iaa.Sequential([
    # crop images by -5% to 10% of their height/width
    sometimes(iaa.CropAndPad(
        percent=(-0.05, 0.1),
        pad_mode=ia.ALL,
        pad_cval=(0, 255)
    )),
    sometimes(iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        # scale images to 80-120% of their size, individually per axis
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        # translate by -20 to +20 percent (per axis)
        rotate=(-20, 20),  # rotate by -20 to +20 degrees
        shear=(-20, 20),  # shear by -20 to +20 degrees
        order=[0, 1],
        # use nearest neighbour or bilinear interpolation (fast)
        cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
        mode=ia.ALL
        # use any of scikit-image's warping modes (see 2nd image from the top for examples)
    )),
    # execute 0 to 5 of the following (less important) augmenters per image
    # don't execute all of them, as that would often be way too strong
    iaa.SomeOf((0, 5),
               [
                   # Create some super-pixels
                   sometimes(iaa.Superpixels(p_replace=(0, 0.5), n_segments=(2, 10))),
                   # Blur the image
                   iaa.OneOf([
                       iaa.GaussianBlur((0, 1.0)),
                       iaa.AverageBlur(k=(1, 7)),
                       iaa.MedianBlur(k=(1, 7)),
                   ]),
                   # Convolutional operations
                   iaa.OneOf([
                       iaa.Sharpen(alpha=(0, 1.0), lightness=(0.7, 1.3)),
                       iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                   ]),
                   iaa.SimplexNoiseAlpha(
                       iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0.5, 1.0)),
                           iaa.DirectedEdgeDetect(alpha=(0.5, 1.0),
                                                  direction=(0.0, 1.0)),
                       ])
                   ),
                   # Add noise to the image
                   iaa.OneOf([
                       iaa.AdditiveGaussianNoise(loc=0,
                                                 scale=(0.0, 0.1 * 255),
                                                 per_channel=0.5),
                       iaa.AdditiveLaplaceNoise(loc=0,
                                                scale=(0.0, 0.1 * 255),
                                                per_channel=0.5),

                   ]),
                   # Dropout some pixels
                   iaa.OneOf([
                       iaa.Dropout((0.01, 0.05), per_channel=0.5),
                       # randomly remove up to 10% of the pixels
                       iaa.CoarseDropout((0.01, 0.1),
                                         size_percent=(0.1, 0.5),
                                         per_channel=0.5),
                   ]),
                   # Play with the colors of the image
                   iaa.OneOf([
                       # Invert color channels
                       iaa.Invert(0.01, per_channel=0.5),
                       # Add hue and saturation
                       iaa.AddToHueAndSaturation((-45, 45)),
                       # Multiply hue and saturation
                       iaa.MultiplyHueAndSaturation((-1, 1))
                   ]),
                   # Change brightness and contrast
                   iaa.OneOf([
                       iaa.Add((-10, 10), per_channel=0.5),
                       iaa.Multiply((0.5, 1.5), per_channel=0.5),
                       iaa.GammaContrast(gamma=(0.5, 1.75), per_channel=0.5),
                       iaa.SigmoidContrast(cutoff=(0, 1), per_channel=0.5),
                       iaa.LogContrast(gain=(0.5, 1), per_channel=0.5),
                       iaa.LinearContrast(alpha=(0.25, 1.75), per_channel=0.5),
                       iaa.HistogramEqualization()
                   ]),
                   sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5),
                                                       sigma=0.25)),
                   # move pixels locally around (with random strengths)
                   sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                   # sometimes move parts of the image around
                   sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.2))),
               ]
               ),
    # With 10 % probability apply one the of the weather conditions
    iaa.Sometimes(0.1, iaa.OneOf([
        iaa.Clouds(),
        iaa.Fog(),
        iaa.Snowflakes()
    ])),
    iaa.JpegCompression((0.3, 1))
])
