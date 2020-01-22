import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags
from tensorflow.python.client import device_lib
from tqdm import tqdm

import constants
import helpers
from data.custom_img_data_generator import Generator
from models import train
from models.recognition import recognizer_all

FLAGS = flags.FLAGS
# Strings
flags.DEFINE_string("dataset_name", '', "Dataset name.")
flags.DEFINE_string("extra", '', "Extra information to save in the training information file")

# Integers
flags.DEFINE_integer("epochs", 20, "Number of epochs for training")
flags.DEFINE_integer("save_freq", 2, "Checkpoints frequency")
flags.DEFINE_integer("batch_size", 256, "Batch size for the training data")
flags.DEFINE_integer("img_res", 50, "Image resolution to use for training")
flags.DEFINE_integer("times", 2, "How many times each model is trained for later average the results")

# Floats
flags.DEFINE_float("lr", 2e-3, "Learning rate")

# Boolean
flags.DEFINE_boolean("use_cosine_lr", True, "Use cosine learning rate scheduler")
flags.DEFINE_boolean("show_plots", False, "Show the plots of train/val loss and lr rate every time a model is trained.")
flags.DEFINE_boolean("simple_aug", False, "Apply only simple data augmentations.")


def train_recognition(model_version):
    """
    Train an image classifier. The images should be saved each in a folder with name the class labels.
    It will save a dictionary that maps each output neuron to the corresponding label as a pickle
    file named 'neuron_to_class_dict.p'. Divides de training set into train and validation sets and uses
    the test set to analyze the accuracy of the model.
    It saves the parameters passed to train the model in a file named 'train_info.txt' in the checkpoints
    directory.
    For the moment it has been checked only with the GTSR dataset.
    """

    img_res = FLAGS.img_res
    gpu_aval = tf.test.is_gpu_available(cuda_only=True)
    model = recognizer_all.Recognizer(FLAGS.dataset_name, img_res=img_res, train=False, model_version=model_version)
    gpus = 0
    if gpu_aval:
        for x in device_lib.list_local_devices():
            if x.device_type == "GPU":
                gpus += 1

    print(constants.C_WARNING, "Are CUDA gpus available? \n\t-",
          (constants.C_OKBLUE + ('Yes, ' + str(gpus)) if gpu_aval
           else constants.C_FAIL + 'No.'), constants.C_ENDC)

    batch_size = FLAGS.batch_size
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.lr)

    if gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        # Here the batch size scales up by number of workers since
        # `tf.data.Dataset.batch` expects the global batch size. Previously we used 'batch_size',
        # and now this is multiplied by the number of workers.
        batch_size *= gpus
        with strategy.scope():
            model.train_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    else:
        model.train_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    train_info = FLAGS.extra.replace('\\n', '\n') + '\n'
    train_info += "Train model: {} For {} epochs and {} as Database. \nParameters used:\n    " \
                  "- Frequency = {}\n    " \
                  "- Batch size = {}\n    " \
                  "- Image resolution = {}\n".format(model.model_description, FLAGS.epochs, FLAGS.dataset_name,
                                                     FLAGS.save_freq, FLAGS.batch_size, FLAGS.img_res)

    print(constants.C_WARNING, FLAGS.extra.replace('\\n', '\n'), constants.C_ENDC)
    if FLAGS.dataset_name:
        ds_directory = constants.LOCAL_DATASET_PATH.format(FLAGS.dataset_name)
        if gpus > 0:
            ds_directory = constants.CLUSTER_DATASET_PATH.format(FLAGS.dataset_name)

        train_directory = ds_directory + "train"

        if FLAGS.simple_aug:
            generator = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1. / 255.0,
                validation_split=0.1,
                rotation_range=20,
                height_shift_range=0.2,
                width_shift_range=0.2,
                zoom_range=0.3,
                shear_range=0.3,
                brightness_range=(0.2, 0.8)
            )
            train_generator = generator.flow_from_directory(
                train_directory,
                target_size=(img_res, img_res),
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )
            val_generator = generator.flow_from_directory(
                train_directory,
                target_size=(img_res, img_res),
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation'
            )
            class_to_neuron_dict = train_generator.class_indices
        else:
            generator = Generator(directory=train_directory,
                                  batch_size=FLAGS.batch_size,
                                  image_dimensions=(img_res, img_res),
                                  validation_split=0.1)
            train_generator = generator.train_generator
            val_generator = generator.val_generator
            class_to_neuron_dict = generator.class_indices

        neuron_to_class_dict = {}
        labels_dict = helpers.get_labels_dict(model.dataset_name)
        neuron_labels = []
        for class_id, neuron in class_to_neuron_dict.items():
            neuron_to_class_dict[str(neuron)] = str(class_id)
            neuron_labels.append(labels_dict[str(neuron)])

        start = time.time()
        history, history_callback = train.train(model=model,
                                                epochs=FLAGS.epochs,
                                                train_data=train_generator,
                                                val_data=val_generator,
                                                save_freq=FLAGS.save_freq,
                                                initial_lr=FLAGS.lr,
                                                train_info=train_info,
                                                use_fit_generator=gpus <= 1,
                                                use_cosine_lr=FLAGS.use_cosine_lr,
                                                save_info=False,
                                                show_plots=FLAGS.show_plots)
        test_acc = 100 * test_model(model, neuron_to_class_dict)
        history_callback.test_acc = test_acc
        print(constants.C_OKGREEN, "Test accuracy {:0.2f} %".format(test_acc),
              constants.C_ENDC)

    else:
        print("Train with fake data")
        train_data, val_data = helpers.load_fake_dataset_recognition(FLAGS.img_res)
        start = time.time()
        history, history_callback = train.train(model, FLAGS.epochs, train_data, val_data, FLAGS.save_freq,
                                                FLAGS.lr, 'Use FAKE DS\n', False, FLAGS.use_cosine_lr, False, False)
        history_callback.test_acc = 0.0

    return 1


def test_model(model, neuron_to_class_dict):
    """
    Test the model and returns the accuracy. It expects a csv file with a Path column and a ClassId.
    Tha path column should have the path of each image and the ClassId column the class of each image,
    this classes have to be the same as the folder names in the training set.
    :param model: Model to test
    :param neuron_to_class_dict: Dictionary to translate from neuron to class
    :return: Accuracy in a range from 0 to 1.
    """
    images_path = constants.DATASET_PATH.format(FLAGS.dataset_name) + 'test/'
    test = pd.read_csv(constants.DATASET_PATH.format(FLAGS.dataset_name) + 'Test.csv')

    test['Path'] = test['Path'].apply(lambda x: x.split('/')[-1])
    test = test.set_index('Path')
    total, wrong = 0, 0
    images = []
    classes = []
    for i, image_id in enumerate(tqdm(os.listdir(images_path))):
        if 'png' not in image_id:
            continue
        image_path = images_path + image_id
        img = tf.image.decode_image(open(image_path, 'rb').read(), channels=3)
        images.append(helpers.transform_images(img, model.img_res))
        real_class_id = test.loc[image_id]['ClassId']
        classes.append(str(real_class_id))
        if i % (FLAGS.batch_size - 1) == 0 and i > 0:
            total += FLAGS.batch_size
            pred_class_id = model.inference_model.predict_classes(tf.convert_to_tensor(images))
            for j, pred_class in enumerate(pred_class_id):
                pred_class = neuron_to_class_dict[str(pred_class)]
                wrong += classes[j] != pred_class
            images, classes = [], []

    return 1 - wrong / total


def main(_argv):
    models = recognizer_all.versions
    times = FLAGS.times
    results = []
    history_callbacks = []
    history_callbacks_df = []
    analysis = []
    for model in models:
        max_train_accs, max_val_accs, epoch_times = [], [], []
        for time in range(times):
            print(constants.C_WARNING, 'Start training the model', model,
                  ('{}/{}'.format(time, times) if times > 1 else ''), constants.C_ENDC)
            _, _, history_callback = train_recognition(model_version=model)
            max_train_acc = np.max(history_callback.train_acc)
            max_train_accs.append(max_train_acc)
            max_val_acc = np.max(history_callback.val_acc)
            max_val_accs.append(max_val_acc)
            epoch_times.append(np.mean(history_callback.times))
            history_callbacks.append(history_callback)
            history_callbacks_df.append(
                [
                    model,
                    max_train_acc,
                    max_val_acc,
                    FLAGS.use_cosine_lr,
                    FLAGS.img_res,
                    FLAGS.lr,
                    history_callback.total_time,
                    history_callback.loss,
                    history_callback.val_loss,
                    history_callback.train_acc,
                    history_callback.val_acc,
                    history_callback.learning_rates,
                    history_callback.times,
                    history_callback.history
                ]
            )
            analysis.append([
                model,
                'cosine' if FLAGS.use_cosine_lr else 'constant',
                FLAGS.img_res,
                FLAGS.lr,
                'simple' if FLAGS.simple_aug else 'v1',
                max_train_acc,
                max_val_acc,
                history_callback.test_acc / 100
            ])
        results.append([100 * np.average(max_train_accs), 100 * np.average(max_val_accs), int(np.mean(epoch_times))])

    history_callbacks_df = pd.DataFrame(history_callbacks_df,
                                        columns=['model', 'max_train_acc', 'max_val_acc', 'use_cosine_lr', 'img_res',
                                                 'initial_lr', 'ttime', 'train loss', 'val loss', 'train acc',
                                                 'val acc', 'lrs', 'times', 'history'])
    analysis = pd.DataFrame(analysis,
                            columns=['model', 'scheduler', 'img_res', 'initial_lr', 'augmentation', 'max_train_acc',
                                     'max_val_acc', 'test_acc'])

    train_info = "Parameters used:\n    " \
                 "- Frequency = {}\n    " \
                 "- Batch size = {}\n    " \
                 "- Dataset = {}\n    " \
                 "- Image resolution = {}\n    " \
                 "- Epochs = {}\n".format(FLAGS.save_freq, FLAGS.batch_size, FLAGS.dataset_name,
                                          FLAGS.img_res, FLAGS.epochs)
    if FLAGS.use_cosine_lr:
        train_info += "    - Use cosine decay scheduler. Initial LR = " + str(FLAGS.lr)
    else:
        train_info += "    - Use constant LR = " + str(FLAGS.lr)

    print()
    print(constants.C_WARNING, train_info, constants.C_ENDC)
    print()
    for i, model in enumerate(models):
        print(constants.C_OKGREEN,
              model,
              'had: max_train_acc = {:0.2f} %, max_val_acc = {:0.2f} %, test accuracy = {:0.2f} %. '
              'Training an epoch took {} on average.'
              .format(results[i][0], results[i][1], history_callbacks[i].test_acc, helpers.display_time(results[i][2])),
              constants.C_ENDC)

    path_all = constants.PROJECT_PATH + 'reports/recognizer_analysis' + '-img_res-' + str(FLAGS.img_res) + (
        '-' if FLAGS.use_cosine_lr else '-no') + 'cosine_lr-' + str(FLAGS.lr) + '.csv'
    path_mini = constants.PROJECT_PATH + 'reports/recognizer_analysis.csv'
    if os.path.exists(path_all):
        history_callbacks_df = pd.read_csv(path_all).append(history_callbacks_df)
    if os.path.exists(path_mini):
        analysis = pd.read_csv(path_mini).append(analysis)
    history_callbacks_df.to_csv(path_all, index=False)
    analysis.to_csv(path_mini, index=False)
    return results


if __name__ == '__main__':
    app.run(main)
