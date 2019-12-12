import datetime
import pickle
import time
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

import constants
import helpers
from visualization import visualize


class History(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.times = []
        self.history = []
        self.total_time, self.epoch_time_start = 0, 0
        self.loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.learning_rates = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        taken_time = int(time.time() - self.epoch_time_start)
        lr = K.get_value(self.model.optimizer.lr)
        self.total_time += taken_time
        time_str = helpers.display_time(int(taken_time))
        if 'accuracy' in logs:
            history = "- Epoch " + str(
                epoch + 1) + " finished in " + time_str + ". Training loss = {:0.2f}" \
                                                          ". Validation loss = {:0.2f}" \
                                                          ". Train accuracy = {:0.2f} %" \
                                                          ". Validation accuracy = {:0.2f} %" \
                                                          ". LR = {:f}".format(
                logs['loss'], logs['val_loss'], 100 * logs['accuracy'], 100 * logs['val_accuracy'], lr)
        else:
            history = "- Epoch " + str(
                epoch + 1) + " finished in " + time_str + " with a training loss of {:0.2f} and " \
                                                          "a validation loss of {:0.2f}. " \
                                                          "LR = {:f}".format(
                logs['loss'], logs['val_loss'], lr)

        print('\n\n \033[94m' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), history + '\033[0m \n\n')
        self.history.append(history)
        self.times.append(taken_time)
        self.loss.append(logs['loss'])
        self.val_loss.append(logs['val_loss'])
        if 'accuracy' in logs:
            self.train_acc.append(logs['accuracy'])
            self.val_acc.append(logs['val_accuracy'])
        self.learning_rates.append(lr)


class CustomSaveHistory(tf.keras.callbacks.Callback):
    def __init__(self, path_to_save_history):
        self.path_to_save_history = path_to_save_history
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        pickle.dump(self.history, open(self.path_to_save_history, "wb"))


class ModelCheckpoint(tf.keras.callbacks.Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled with the values of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_freq: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', save_freq=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.save_freq:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


class CosineDecayScheduler(tf.keras.callbacks.Callback):
    """ Cosine decay learning rate scheduler """

    def __init__(self, initial_lr: float, epochs: int, global_epoch_init: int = 0, epochs_hold_initial_lr: int = 0,
                 minimum_lr: float = 1e-5, verbose: int = 0):
        """
        Constructor for cosine decay with warmup learning rate scheduler.

        :param initial_lr: Initial learning rate.
        :param epochs: Number of epochs.
        :param global_epoch_init: initial epoch, e.g. from previous checkpoint.
        :param epochs_hold_initial_lr: Number of epochs to hold the initial learning rate before decaying.
        :param minimum_lr: Minimum learning rate to use.
        :param verbose: 0: quiet, 1: update messages.
        """

        self.initial_lr = initial_lr
        self.total_steps = epochs
        self.global_epoch = global_epoch_init
        self.epochs_hold_initial_lr = epochs_hold_initial_lr
        self.verbose = verbose
        self.learning_rates = []
        self.minimum_lr = minimum_lr

    def on_epoch_end(self, epoch, logs=None):
        self.global_epoch = self.global_epoch + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.epochs_hold_initial_lr:
            learning_rate = self.initial_lr
        else:
            learning_rate = 0.5 * self.initial_lr * (1 + np.cos(
                np.pi * (self.global_epoch - self.epochs_hold_initial_lr)
                / float(self.total_steps - self.epochs_hold_initial_lr)))
        if learning_rate < self.minimum_lr:
            learning_rate += self.minimum_lr

        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('\nEpoch {}: setting learning rate to {}.'.format(self.global_epoch + 1, learning_rate))


class TensorBoardImagesDetection(tf.keras.callbacks.Callback):
    """ Add images to the tensorboard logs """

    def __init__(self, inference_model, tfrecords_pattern_path: str, dataset_name: str, model_input_size: int,
                 freq: int, logs_path: str, n_images=10):
        """
        Constructor of the Callback to add images to the tensorboard logs. It will select images randomly from the
        tfrecords.

        :param inference_model: Model to use for the inference.
        :param tfrecords_pattern_path: Path to the tfrecords to extract the images to show.
        :param dataset_name: Name of the dataset used for training.
        :param model_input_size: Size of the input of the model.
        :param freq: Frequency to execute the images.
        :param logs_path: Path to save the logs.
        :param n_images: Number of images to use.
        """

        self.inference_model = inference_model
        self.images, self.images_transformed, self.images_shape, self.groundtruth_ann = [], [], [], []
        self.save_freq = freq
        self.epochs_since_last_save = 0
        self.dataset_name = dataset_name
        self.logs_path = logs_path
        files = tf.random.shuffle(tf.io.matching_files(tfrecords_pattern_path))
        shards = tf.data.Dataset.from_tensor_slices(files)
        dataset = shards.interleave(tf.data.TFRecordDataset)
        dataset = dataset.shuffle(buffer_size=800)
        for tfrecord in dataset.take(n_images):
            x = tf.io.parse_single_example(tfrecord, constants.IMAGE_FEATURE_MAP)
            image = tf.image.decode_jpeg(x['image/encoded'], channels=3).numpy()
            groundtruth_ann = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                                        tf.sparse.to_dense(x['image/object/class/label'])], axis=1).numpy()
            groundtruth_ann[:, :-1] = groundtruth_ann[:, :-1] * np.array(
                [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])

            self.groundtruth_ann.append(groundtruth_ann.astype(np.int32).tolist())
            self.images.append(image)
            self.images_transformed.append(helpers.transform_images(image, model_input_size))
            self.images_shape.append(np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]]))

        self.images_transformed = tf.convert_to_tensor(self.images_transformed)

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save < self.save_freq:
            return

        self.epochs_since_last_save = 0
        images_annotated = []
        images_cached = []
        cache_size = 8
        for img_idx, image_transformed in enumerate(self.images_transformed):
            images_cached.append(image_transformed)
            if img_idx > 0 and (img_idx % cache_size == 0 or img_idx == len(self.images_transformed) - 1):
                bboxes, scores, classes, _ = self.inference_model(tf.convert_to_tensor(images_cached))
                for j in range(bboxes.shape[0]):
                    real_img_idx = img_idx - cache_size + j
                    img_boxes, img_scores, img_classes = [np.array(elem[j]) for elem in [bboxes, scores, classes]]
                    annotations = []

                    for object_index in range(len(img_scores)):
                        bbox = [int(elem) for elem in list(img_boxes[object_index] * self.images_shape[real_img_idx])]
                        annotations.append(bbox + [int(img_classes[object_index])])

                    images_annotated.append(
                        visualize.load_tensorboard_image(self.images[real_img_idx], self.groundtruth_ann[real_img_idx],
                                                         annotations, img_scores, self.dataset_name))
                images_cached = []

        file_writer = tf.summary.create_file_writer(self.logs_path)

        # Using the file writer, log the reshaped image.
        with file_writer.as_default():
            tf.summary.image(name='Groundtruth in the left image. Prediction on the right image.',
                             data=tf.convert_to_tensor(images_annotated),
                             step=epoch + 1,
                             max_outputs=len(images_annotated))
