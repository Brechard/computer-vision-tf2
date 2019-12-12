import time

import tensorflow as tf
from absl import app, flags
from tensorflow.python.client import device_lib

import constants
import helpers
from data.dataset import Dataset
from models import custom_callbacks
from models import train
from models.detection.yolov3 import YOLOv3

FLAGS = flags.FLAGS
# Strings
flags.DEFINE_string("dataset_name", '', "Dataset name.")
flags.DEFINE_string("extra", '', "Extra information to save in the training information file")
# ['none', 'all', 'features', 'last_block', 'last_conv']
trainable = ['none', constants.TRAINABLE_ALL, constants.TRAINABLE_FEATURES, constants.TRAINABLE_LAST_BLOCK,
             constants.TRAINABLE_LAST_CONV]
# Enum
flags.DEFINE_enum("trainable", 'none', trainable,
                  "Use transfer learning from the original weights and keep some part of the network trainable."
                  "none: do not use transfer learning"
                  "all: all the model is trainable"
                  "features: freeze the feature extractor (like DarkNet) and the rest is trainable"
                  "last_block: only the last block of layers is trainable"
                  "last_conv: only the last conv layer is trainable")

# Integers
flags.DEFINE_integer("epochs", 100, "Number of epochs for training")
flags.DEFINE_integer("save_freq", 5, "Checkpoints frequency")
flags.DEFINE_integer("batch_size", 32, "Batch size for the training data")

# Floats
flags.DEFINE_float("lr", 2e-3, "Learning rate")

# Boolean
flags.DEFINE_boolean("tiny", False, "Flag to use tiny version of YOLO")
flags.DEFINE_boolean("use_cosine_lr", True, "Use cosine learning rate scheduler")


def train_detection(_argv):
    gpu_aval = tf.test.is_gpu_available(cuda_only=True)
    gpus = 0
    if gpu_aval:
        for x in device_lib.list_local_devices():
            if x.device_type == "GPU":
                gpus += 1

    print(constants.C_WARNING, "Are CUDA gpus available? \n\t-",
          (constants.C_OKBLUE + ('Yes, ' + str(gpus)) if gpu_aval
           else constants.C_FAIL + 'No.'), constants.C_ENDC)

    batch_size = FLAGS.batch_size
    if gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        # Here the batch size scales up by number of workers since
        # `tf.data.Dataset.batch` expects the global batch size. Previously we used 'batch_size',
        # and now this is multiplied by the number of workers.
        batch_size *= gpus
        with strategy.scope():
            dataset, model = load_model_and_db(batch_size)
    else:
        dataset, model = load_model_and_db(FLAGS.batch_size)

    if FLAGS.dataset_name == '':
        tfrecords_pattern_path = constants.PROCESSED_PROJECT_FOLDER_PATH + \
                                 constants.TFRECORDS_PATH.format('COCO', 'val', '*-of-*')
        tensorboard_imgs = custom_callbacks.TensorBoardImagesDetection(
            inference_model=model.inference_model,
            tfrecords_pattern_path=tfrecords_pattern_path,
            dataset_name='COCO',
            model_input_size=model.image_res,
            freq=FLAGS.save_freq,
            logs_path=model.logs_path,
            n_images=5)
        start = time.time()
        history, history_callback = train.train(model, FLAGS.epochs, dataset.train_data, dataset.validation_data,
                                                FLAGS.save_freq, FLAGS.lr, 'Use fake DS\n', False, True,
                                                extra_callbacks=[tensorboard_imgs])
        helpers.save_history(FLAGS, model.model_name, dataset.dataset_name, history, start, 'detection')
        return model, history

    train_info = FLAGS.extra.replace('\\n', '\n') + '\n'
    train_info += "Train model: {} For {} epochs and {} as Database. \nParameters used:\n    " \
                  "- Checkpoints frequency = {}\n    " \
                  "- Batch size = {}\n".format(model.model_name, FLAGS.epochs, FLAGS.dataset_name,
                                               FLAGS.save_freq, FLAGS.batch_size)
    if gpu_aval:
        train_info += "    - {} gpu{} available for training\n".format(gpus, 's' if gpus > 1 else '')
    train_info += "    - Use {} version of the model\n".format('tiny' if FLAGS.tiny else 'full')
    if FLAGS.trainable != 'none':
        train_info += "    - Use transfer learning with trainable option: {} \n".format(FLAGS.trainable)
    else:
        train_info += "    - Train from scratch\n"

    print(constants.C_WARNING, FLAGS.extra.replace('\\n', '\n'), constants.C_ENDC)

    tfrecords_pattern_path = dataset.tf_paths.format(dataset.dataset_name, 'val', '*-of-*')
    tensorboard_imgs = custom_callbacks.TensorBoardImagesDetection(inference_model=model.inference_model,
                                                                   tfrecords_pattern_path=tfrecords_pattern_path,
                                                                   dataset_name=dataset.dataset_name,
                                                                   model_input_size=model.image_res,
                                                                   freq=FLAGS.save_freq,
                                                                   logs_path=model.logs_path,
                                                                   n_images=10)
    start = time.time()
    history, history_callback = train.train(model=model,
                                            epochs=FLAGS.epochs,
                                            train_data=dataset.train_data,
                                            val_data=dataset.validation_data,
                                            save_freq=FLAGS.save_freq,
                                            initial_lr=FLAGS.lr,
                                            train_info=train_info,
                                            use_fit_generator=False,
                                            use_cosine_lr=FLAGS.use_cosine_lr,
                                            extra_callbacks=[tensorboard_imgs])

    helpers.save_history(FLAGS, model.model_name, dataset.dataset_name, history, start, 'detection')
    return model, history


def load_model_and_db(batch_size):
    model = YOLOv3(tiny=FLAGS.tiny)
    dataset = Dataset(FLAGS.dataset_name, FLAGS.tiny)
    dataset.load_datasets(model.image_res, model.anchors, model.masks, batch_size)
    # When loading the model, the folders to save the checkpoints, figures and logs are created.
    if FLAGS.trainable == 'none':
        model.load_models(dataset=dataset,
                          for_training=True,
                          plot_model=False)
    else:
        model.load_for_transfer_learning(dataset, trainable_option=FLAGS.trainable)
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.lr)
    # model.train_model.load_weights(
    #     '/home/brechard/models_results/testing/20191017_180805_COCO/checkpoints/YOLOv3_final.ckpt')
    loss = model.get_loss()
    model.train_model.compile(optimizer=optimizer, loss=loss,
                              run_eagerly=False, metrics=['accuracy'])

    return dataset, model


if __name__ == '__main__':
    app.run(train_detection)
