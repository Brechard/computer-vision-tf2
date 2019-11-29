import numpy as np
import tensorflow as tf
from absl import app, flags

import constants
import helpers
import visualization.visualize as visualize
from data.dataset import Dataset
from models.detection.yolov3 import YOLOv3

FLAGS = flags.FLAGS
# Strings
flags.DEFINE_string("dataset_name", 'COCO', "Dataset name. Used for translating from label id to label text.")
flags.DEFINE_string("img_path", constants.PROJECT_PATH + 'reports/test_image.jpg', "Image path.")
flags.DEFINE_string("output_path", constants.PROJECT_PATH + 'reports/figures/test_image_out.png', "Image path.")
flags.DEFINE_string("weights_path", '', "Path to the weights file.")
flags.DEFINE_string("title", '', 'title to plot in the image')

# Booleans
flags.DEFINE_boolean("tiny", False, 'Use the full or the tiny model.')


def detect(model, dataset_name: str, image_path: str, title: str = "", output_path: str = None):
    labels_map_dict = helpers.get_labels_dict(dataset_name)

    img = tf.image.decode_image(open(image_path, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img_shape = np.array([img.shape[2], img.shape[1], img.shape[2], img.shape[1]])

    img = helpers.transform_images(img, model.image_res)
    boxes, scores, classes, nums = model.inference_model(img)
    boxes, scores, classes = [np.array(elem[0]) for elem in [boxes, scores, classes]]
    annotations = []

    for object_index in range(nums[0]):
        bbox = [int(elem) for elem in list(boxes[object_index] * img_shape)]
        annotations.append(bbox + [int(classes[object_index])])
        print(constants.C_OKGREEN, 'Object', labels_map_dict[str(int(classes[object_index]))],
              'with probability', scores[object_index],
              'is in', annotations[object_index][:-1], constants.C_ENDC)
    visualize.plot_image_annotations(image_path, annotations, labels_map_dict, scores, title, output_path)
    if len(annotations) == 0:
        print(constants.C_WARNING, "No object found.", title, constants.C_ENDC)


def main(_argv):
    model = YOLOv3(tiny=FLAGS.tiny)
    if FLAGS.weights_path:
        model.load_models(Dataset(FLAGS.dataset_name), for_training=False)
        model.inference_model.load_weights(FLAGS.weights_path)
        print(constants.C_OKBLUE, "Weights from", FLAGS.weights_path, 'loaded successfully', constants.C_ENDC)
    else:
        model.load_original_yolov3()
        print(constants.C_OKBLUE, "Successfully loaded weights from the original paper", constants.C_ENDC)

    detect(model, FLAGS.dataset_name, FLAGS.img_path, FLAGS.title, FLAGS.output_path)


if __name__ == '__main__':
    app.run(main)
