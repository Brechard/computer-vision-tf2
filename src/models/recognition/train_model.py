import json
import os
import time

import pandas as pd
import tensorflow as tf
from absl import app, flags
from tensorflow.python.client import device_lib
from tqdm import tqdm

import constants
import helpers
from models import train
from models.recognition.recognizer import Recognizer

FLAGS = flags.FLAGS
# Strings
flags.DEFINE_string("model_name", "Recognizer", "Dataset name.")
flags.DEFINE_string("dataset_name", 'GTSR', "Dataset name.")
flags.DEFINE_string("extra", '', "Extra information to save in the training information file")

# Integers
flags.DEFINE_integer("epochs", 20, "Number of epochs for training")
flags.DEFINE_integer("save_freq", 5, "Checkpoints frequency")
flags.DEFINE_integer("batch_size", 256, "Batch size for the training data")
flags.DEFINE_integer("img_res", 50, "Image resolution to use for training")

# Floats
flags.DEFINE_float("lr", 2e-3, "Learning rate")

# Boolean
flags.DEFINE_boolean("use_cosine_lr", True, "Use cosine learning rate scheduler")


def train_recognition(_argv):
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
    model = Recognizer(FLAGS.dataset_name, img_res=img_res, train=True, model_name=FLAGS.model_name)
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

    model.train_model.summary()

    train_info = FLAGS.extra.replace('\\n', '\n') + '\n'
    train_info += "Train model: {} For {} epochs and {} as Database. \nParameters used:\n    " \
                  "- Frequency = {}\n    " \
                  "- Batch size = {}\n    " \
                  "- Image resolution = {}\n".format(model.model_description, FLAGS.epochs, FLAGS.dataset_name,
                                                     FLAGS.save_freq, FLAGS.batch_size, FLAGS.img_res)

    print(constants.C_WARNING, FLAGS.extra.replace('\\n', '\n'), constants.C_ENDC)
    if model.dataset_name != 'FAKE':
        ds_directory = constants.DATASET_PATH.format(FLAGS.dataset_name)
        train_directory = ds_directory + "train"

        generator = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1. / 255.0,
            validation_split=0.1,
            rotation_range=20,
            height_shift_range=0.2,
            width_shift_range=0.2,
            zoom_range=0.1,
            shear_range=0.1,
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
        neuron_to_class_dict = {}
        for k, v in class_to_neuron_dict.items():
            neuron_to_class_dict[str(v)] = str(k)

        print('train_generator.class_indices', train_generator.class_indices)
        print('val_generator.class_indices', val_generator.class_indices)
        start = time.time()
        history, history_callback = train.train(model=model,
                                                epochs=FLAGS.epochs,
                                                train_data=train_generator,
                                                val_data=val_generator,
                                                save_freq=FLAGS.save_freq,
                                                initial_lr=FLAGS.lr,
                                                train_info=train_info,
                                                use_fit_generator=gpus <= 1,
                                                use_cosine_lr=FLAGS.use_cosine_lr)

        model.train_model.save_weights(model.checkpoints_path + 'weights.ckpt')

        with open(model.checkpoints_path + 'neuron_to_class_dict.json', 'w') as fp:
            json.dump(neuron_to_class_dict, fp, indent=4)

        test_acc = 100 * test_model(model, neuron_to_class_dict)
        history_callback.test_acc = test_acc
        print(constants.C_OKGREEN, "Test accuracy {:0.2f} %".format(test_acc),
              constants.C_ENDC)

        with open(model.checkpoints_path + 'train_info.txt', 'a') as t:
            t.write("Test accuracy {:0.2f} %".format(test_acc))

    else:
        print("Train with fake data")
        train_data, val_data = helpers.load_fake_dataset_recognition()
        start = time.time()
        history, history_callback = train.train(model, FLAGS.epochs, train_data, val_data, FLAGS.save_freq,
                                                FLAGS.lr, 'Use FAKE DS\n', False, FLAGS.use_cosine_lr)

    helpers.save_history(FLAGS, model.model_name, model.dataset_name, history, start, 'recognition')
    return model, history, history_callback


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


if __name__ == '__main__':
    app.run(train_recognition)
