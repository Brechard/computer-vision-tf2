import json

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import app, flags

import helpers
from models.recognition.recognizer import Recognizer

FLAGS = flags.FLAGS
# Strings
flags.DEFINE_string("dir_path", '', "Directory path that should contain the weights of the model and the json file"
                                    "that translates from neuron to class")
flags.DEFINE_string("img_path", '', "Image path to classify.")
flags.DEFINE_string("dataset_name", 'GTSR', "Dataset name. Used for translating from label id to label text.")

# Integers
flags.DEFINE_integer("img_res", 50, "Image resolution used when training.")


def classify_image(dir_path: str, img_path: str, dataset_name: str, img_res: int):
    """
    Classify the given image with the given model.
    In the model directory there must to be two files:
        1. Weights of the model. Named: 'weights.ckpt'
        2. Dictionary in a json to translate from neuron to class. Named: 'neuron_to_class_dict.json'

    :param dir_path: Path to the directory
    :param img_path: Path to the image
    :param dataset_name: Dataset that the image belongs to.
    :param img_res: Image size in pixels used to train the model.
    """
    with open(dir_path + 'neuron_to_class_dict.json', "r") as f:
        neuron_to_class_dict = json.load(f)

    model = Recognizer(dataset_name, img_res=img_res, train=False)
    model.inference_model.load_weights(dir_path + 'weights.ckpt')
    img = tf.image.decode_image(open(img_path, 'rb').read(), channels=3)
    img = tf.expand_dims(helpers.transform_images(img, model.img_res), 0)
    img_classes = model.inference_model(img)
    img_class_prob = np.max(img_classes.numpy())
    img_class_id = np.argmax(img_classes.numpy())
    img_class_id = neuron_to_class_dict[str(img_class_id)]

    plt.imshow(img[0])
    plt.title('{:0.2f} % '.format(img_class_prob * 100) + model.labels_map_dict[img_class_id].upper())
    plt.show()


def main(_argv):
    classify_image(dir_path=FLAGS.dir_path, img_path=FLAGS.img_path, dataset_name=FLAGS.dataset_name,
                   img_res=FLAGS.img_res)


if __name__ == '__main__':
    app.run(main)
