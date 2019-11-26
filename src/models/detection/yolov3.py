from os.path import exists

import numpy as np
import tensorflow.keras.layers as layers
from absl import logging
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.utils import plot_model

import constants
import helpers
from data.dataset import Dataset
from models.detection.yolo_layers import *

"""
Every layer has to know about all of the anchor boxes but is only predicting some subset of them. 
This could probably be named something better but the mask tells the layer which of the bounding boxes 
it is responsible for predicting. The first yolo layer predicts 6,7,8 because those are the largest boxes
and it's at the coarsest scale. The 2nd yolo layer predicts some smaller ones, etc.
"""


def darknet(model_input, use_plot_model: bool = False, figs_path: str = None):
    x = conv_2d(model_input, n_filters=32, kernel_size=3, stride=1)
    x = conv_2d(x, n_filters=64, kernel_size=3, stride=2)
    x = darknet_residual_block(x, n_filters_list=[32, 64], kernel_sizes_list=[1, 3], n_blocks=1)
    x = conv_2d(x, n_filters=128, kernel_size=3, stride=2)
    x = darknet_residual_block(x, n_filters_list=[64, 128], kernel_sizes_list=[1, 3], n_blocks=2)
    x = conv_2d(x, n_filters=256, kernel_size=3, stride=2)
    x = small_features = darknet_residual_block(x, n_filters_list=[128, 256], kernel_sizes_list=[1, 3], n_blocks=8)
    x = conv_2d(x, n_filters=512, kernel_size=3, stride=2)
    x = medium_features = darknet_residual_block(x, n_filters_list=[256, 512], kernel_sizes_list=[1, 3], n_blocks=8)
    x = conv_2d(x, n_filters=1024, kernel_size=3, stride=2)
    big_features = darknet_residual_block(x, n_filters_list=[512, 1024], kernel_sizes_list=[1, 3], n_blocks=4)
    darknet_model = tf.keras.Model(model_input, (big_features, medium_features, small_features), name="DarkNet")
    if use_plot_model:
        plot_model(darknet_model, to_file=figs_path + 'darknet.png', show_shapes=True, show_layer_names=True)
    return darknet_model(model_input)


def tiny_darknet(model_input, use_plot_model: bool = False, figs_path: str = None):
    x = conv_2d(model_input, n_filters=16, kernel_size=3, stride=1, max_pool=True)
    x = conv_2d(x, n_filters=32, kernel_size=3, stride=1, max_pool=True)
    x = conv_2d(x, n_filters=64, kernel_size=3, stride=1, max_pool=True)
    x = conv_2d(x, n_filters=128, kernel_size=3, stride=1, max_pool=True)
    x = small_features = conv_2d(x, n_filters=256, kernel_size=3, stride=1, max_pool=False)
    x = layers.MaxPool2D(2, 2, 'same')(x)
    x = conv_2d(x, n_filters=512, kernel_size=3, stride=1, max_pool=False)
    x = layers.MaxPool2D(2, 1, 'same')(x)
    big_features = conv_2d(x, n_filters=1024, kernel_size=3, stride=1, max_pool=False)
    tiny_darknet_model = tf.keras.Model(model_input, (big_features, small_features), name="Tiny_DarkNet")
    if use_plot_model:
        plot_model(tiny_darknet_model, to_file=figs_path + 'tiny_darknet.png', show_shapes=True, show_layer_names=True)
    return tiny_darknet_model(model_input)


class YOLOv3:
    def __init__(self, anchors: list = None,
                 masks: list = None, image_res: int = constants.YOLO_DEFAULT_IMAGE_RES,
                 tiny: bool = False, iou_threshold: float = 0.5,
                 score_threshold: float = 0.5):
        """
        Create a YOLOv3 Network!
        :param anchors: Normalized list with the anchors
        :param masks: Matrix with the indexes of the anchors
        :param image_res: Resolution of images
        :param tiny: Flag to use the tiny version of the network or not
        :param iou_threshold: Threshold to consider a IoU valid
        :param score_threshold: Threshold for objecteness score
        """
        assert image_res % 32 == 0
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.image_res = image_res
        self.tiny = tiny
        if tiny:
            self.model_name = constants.Tiny_YOLOv3
            self.anchors = constants.TINY_YOLO_DEFAULT_ANCHORS if anchors is None else anchors
            self.masks = constants.TINY_YOLO_DEFAULT_ANCHOR_MASKS if masks is None else masks
        else:
            self.model_name = constants.YOLOv3
            self.anchors = constants.YOLO_DEFAULT_ANCHORS if anchors is None else anchors
            self.masks = constants.YOLO_DEFAULT_ANCHOR_MASKS if masks is None else masks

        self.n_classes, self.train_model, self.inference_model = None, None, None
        self.checkpoints_path, self.logs_path, self.figs_path = None, None, None
        if np.max(self.anchors) > 1:
            raise Exception("The anchors must be normalized")

    def create_models(self, use_plot_model: bool):
        """ Standard YOLOv3 implementation, returns the trainable model and the inference model """
        model_input = layers.Input([self.image_res, self.image_res, 3])
        if self.model_name == constants.Tiny_YOLOv3:
            train_model, inference_model = self.tiny_yolo_v3(model_input, use_plot_model)
        else:
            train_model, inference_model = self.yolo_v3(model_input, use_plot_model)

        print(constants.C_OKBLUE, "Model", self.model_name, "created", constants.C_ENDC)
        return train_model, inference_model

    def extract_from_predictions(self, predictions: tf.Tensor, scale_index: int):
        """
        Code from: https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/dataset.py
        Given the prediction in a tensor (batch, grid, grid, anchors, (x, y, w, h, obj, ...classes))
        calculate the bounding box (bbox) position and retrieve the objectness score, the class probabilities
        and the bbox to calculate the loss
        :param predictions: tensor with the prediction
        :param scale_index: index to get the anchors
        :return: bbox, objectness, class_probs, pred_bbox
        """
        # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
        grid_size = tf.shape(predictions)[1]
        box_xy, box_wh, objectness, class_probs = tf.split(
            predictions, (2, 2, 1, self.n_classes), axis=-1)

        # Apply sigmoid function as said in the paper
        box_xy = tf.sigmoid(box_xy)
        objectness = tf.sigmoid(objectness)
        class_probs = tf.sigmoid(class_probs)
        pred_bbox = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

        # !!! grid[x][y] == (y, x)
        grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

        # Right now each of the positions is relative to their grid but we want them relative to the whole image
        box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
        box_wh = tf.exp(box_wh) * self.anchors[self.masks[scale_index]]

        # We calculated the position of the center of the bbox but we need the corners position
        box_x1y1 = box_xy - box_wh / 2
        box_x2y2 = box_xy + box_wh / 2
        bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

        return bbox, objectness, class_probs, pred_bbox

    def get_loss(self):
        if self.model_name == constants.Tiny_YOLOv3:
            sizes = [0, 1]
        else:
            sizes = [0, 1, 2]

        return [self.loss(scale_index) for scale_index in sizes]

    def load_for_transfer_learning(self, dataset: Dataset, trainable_option: str, save_weights_path: str = None,
                                   save_weights: bool = True, for_training=True):
        """
        Load the weights from the original yolov3 model published in https://pjreddie.com/media/files/yolov3.weights
        for transfer learning.
        :param dataset: Dataset to use in the training
        :param trainable_option: Flag to indicate the layers that are trainable. Options are:
        "all", "features", "last_block", "last_conv". It is recommended to use the variables in the constants file.
        :param save_weights_path: Path to save the weights.
        :param save_weights: Flag to save the weights.
        :param for_training: Flag to indicate if the model will be for training and therefore, the folders
        to save the model should be created.
        """
        print(constants.C_OKBLUE, trainable_option, 'layers are trainable', constants.C_ENDC)
        self.load_models(dataset=dataset, for_training=for_training)
        if save_weights_path is None:
            save_weights_path = constants.PROJECT_PATH + 'models/YOLOv3/trainable_' + trainable_option + '_weights' + (
                '-tiny' if self.tiny else '') + '.ckpt'
        if not save_weights:
            save_weights_path = None
        process_original_weights(self.train_model, save_weights_path, self.tiny, trainable_option, False)

    def load_models(self, dataset: Dataset, for_training: bool, plot_model: bool = False):
        if for_training:
            self.checkpoints_path, self.logs_path, self.figs_path = helpers.model_directories(constants.YOLOv3,
                                                                                              dataset.dataset_name)
        else:
            self.figs_path = constants.PROJECT_PATH + 'docs/figures/'

        self.n_classes = dataset.n_classes
        self.train_model, self.inference_model = self.create_models(use_plot_model=plot_model)
        print(constants.C_OKBLUE, "Model", self.model_name, "loaded", constants.C_ENDC)

    def load_original_yolov3(self, for_training=False):
        """
        Load the weights from the original yolov3 model published in https://pjreddie.com/media/files/yolov3.weights
        :param for_training: flag to indicate if the model is for training or not
        """
        self.load_models(dataset=Dataset(constants.COCO), for_training=for_training)
        original_weights_path = constants.PROJECT_PATH + 'models/YOLOv3/original_weights' + (
            '-tiny' if self.tiny else '') + '.ckpt'
        if not exists(original_weights_path + '.index'):
            process_original_weights(self.train_model, original_weights_path, self.tiny, constants.TRAINABLE_ALL)
        self.train_model.load_weights(original_weights_path)
        self.inference_model.load_weights(original_weights_path)

    def loss(self, scale_index):
        def yolo_loss(y_true, y_pred):
            """ Code from: https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/dataset.py """
            # 1. transform all pred outputs
            # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
            pred_box, pred_obj, pred_class, pred_xywh = self.extract_from_predictions(y_pred, scale_index)
            pred_xy = pred_xywh[..., 0:2]
            pred_wh = pred_xywh[..., 2:4]

            # 2. transform all true outputs
            # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
            true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
            true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
            true_wh = true_box[..., 2:4] - true_box[..., 0:2]

            # give higher weights to small boxes
            box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

            # 3. inverting the pred box equations
            grid_size = tf.shape(y_true)[1]
            grid = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
            grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
            true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
            true_wh = tf.math.log(true_wh / self.anchors[self.masks[scale_index]])
            true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)

            # 4. calculate all masks
            obj_mask = tf.squeeze(true_obj, -1)
            # ignore false positive when iou is over threshold
            true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
            best_iou = tf.reduce_max(helpers.broadcast_iou(pred_box, true_box_flat), axis=-1)
            ignore_mask = tf.cast(best_iou < self.iou_threshold, tf.float32)

            # 5. calculate all losses
            xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
            wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
            obj_loss = binary_crossentropy(true_obj, pred_obj)
            obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
            class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)

            # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
            xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
            wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
            obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
            class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))

            return xy_loss + wh_loss + obj_loss + class_loss

        return yolo_loss

    def non_max_suppression(self, outputs):
        """ Code from: https://github.com/zzh8829/yolov3-tf2/blob/master/yolov3_tf2/dataset.py """
        bboxes, confidence, class_prob = [], [], []
        for o in outputs:
            bboxes.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
            confidence.append(tf.reshape(o[1], (tf.shape(o[1])[0], -1, tf.shape(o[1])[-1])))
            class_prob.append(tf.reshape(o[2], (tf.shape(o[2])[0], -1, tf.shape(o[2])[-1])))

        bbox = tf.concat(bboxes, axis=1)
        confidence = tf.concat(confidence, axis=1)
        class_probs = tf.concat(class_prob, axis=1)

        scores = confidence * class_probs
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
            scores=tf.reshape(
                scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
            max_output_size_per_class=100,
            max_total_size=100,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold
        )

        return boxes, scores, classes, valid_detections

    def save_plot_model(self, inference_model, train_model, model):
        plot_model(train_model, to_file=self.figs_path + model + '.png', show_shapes=True, show_layer_names=True)
        plot_model(inference_model, to_file=self.figs_path + model + '_inference.png', show_shapes=True,
                   show_layer_names=True)
        plot_model(train_model, to_file=self.figs_path + model + '_expanded.png', show_shapes=True,
                   show_layer_names=True, expand_nested=True)

    def tiny_yolo_v3(self, model_input, use_plot_model):
        x_big_features, x_small_features = tiny_darknet(model_input, use_plot_model, self.figs_path)

        n_anchors = self.masks.shape[1]

        y_big_features = tiny_layer(x_big_features)
        y_big = last_tiny_layers(y_big_features, n_filters=512, kernel_size=3, n_anchors=n_anchors,
                                 n_classes=self.n_classes)

        concat = concat_layers(128, y_big_features, x_small_features)
        y_small = last_tiny_layers(concat, n_filters=256, kernel_size=3, n_anchors=n_anchors, n_classes=self.n_classes)

        train_model = tf.keras.Model(model_input, (y_big, y_small), name="Tiny_YOLOv3_train")

        boxes_big = layers.Lambda(lambda predictions: self.extract_from_predictions(predictions, 0),
                                  name='extractor_big')(y_big)
        boxes_small = layers.Lambda(lambda predictions: self.extract_from_predictions(predictions, 1),
                                    name='extractor_small')(y_small)
        outputs = layers.Lambda(lambda predictions: self.non_max_suppression(predictions),
                                name='non_max_suppression')((boxes_big[:3], boxes_small[:3]))
        inference_model = tf.keras.Model(model_input, outputs, name='Tiny_YOLOv3_inference')
        if use_plot_model:
            self.save_plot_model(inference_model, train_model, 'tiny-yolov3')
        return train_model, inference_model

    def yolo_v3(self, model_input, use_plot_model):
        x_big_features, x_medium_features, x_small_features = darknet(model_input, use_plot_model, self.figs_path)

        n_anchors = self.masks.shape[1]
        # Block for detecting big objects
        y_big, y_big_features = last_layers(x_big_features, [512, 1024], [1, 3], n_anchors, self.n_classes)
        # Block for detecting medium objects
        concat = concat_layers(256, y_big_features, x_medium_features)
        y_medium, y_medium_features = last_layers(concat, [256, 512], [1, 3], n_anchors, self.n_classes)
        # Block for detecting small objects
        concat = concat_layers(128, y_medium_features, x_small_features)
        y_small, _ = last_layers(concat, [128, 256], [1, 3], len(self.masks), self.n_classes)

        train_model = tf.keras.Model(model_input, (y_big, y_medium, y_small), name="YOLOv3_train")

        boxes_big = layers.Lambda(lambda predictions: self.extract_from_predictions(predictions, 0),
                                  name='extractor_big')(y_big)
        boxes_medium = layers.Lambda(lambda predictions: self.extract_from_predictions(predictions, 1),
                                     name='extractor_medium')(y_medium)
        boxes_small = layers.Lambda(lambda predictions: self.extract_from_predictions(predictions, 2),
                                    name='extractor_small')(y_small)
        outputs = layers.Lambda(lambda predictions: self.non_max_suppression(predictions),
                                name='non_max_suppression')((boxes_big[:3], boxes_medium[:3], boxes_small[:3]))
        inference_model = tf.keras.Model(model_input, outputs, name='YOLOv3_inference')
        if use_plot_model:
            self.save_plot_model(inference_model, train_model, 'yolov3')
        return train_model, inference_model


def process_original_weights(model, save_weights_path, tiny, trainable_option, load_last_conv=True):
    """
    Read the original weights from the paper and takes into consideration if the last layer has to be loaded
    or not. When we do transfer learning, very likely the new dataset has a different number of output,
    therefore it is not possible to load the same weights as from the original paper. Nonetheless,
    those weights have to be read (but not loaded) from the text file so that the other layers can get their weights
    correctly.
    :param model: Model to load the weights on.
    :param save_weights_path: Path where to save the weights.
    :param tiny: Is the model tiny or not?
    :param trainable_option: Flag to indicate the layers that are trainable. Options are:
    "all", "features", "last_block", "last_conv". It is recommended to use the variables in the constants file.
    :param load_last_conv: Flag to load the last convolutional layer. Set to true only when used with COCO.
    """
    wf = open(constants.PROJECT_PATH + 'models/YOLOv3/yolov3' + ('-tiny' if tiny else '') + '.weights', 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    # The first layer is an input layer so there are no weights to load
    model_blocks = [l.name for l in model.layers[1:]]
    if tiny:
        model_blocks = ['Tiny_DarkNet', 'tiny_layer', 'last_layers_512', 'Concatenate_128', 'last_layers_256']
    for i, block in enumerate(model_blocks):
        sub_model = model.get_layer(block)
        if type(sub_model) != tf.keras.Model:
            continue
        trainable = False
        if trainable_option == constants.TRAINABLE_ALL:
            trainable = True
        elif 'DarkNet' not in block and trainable_option == constants.TRAINABLE_FEATURES:
            trainable = True
        elif 'last_layers' in block and trainable_option == constants.TRAINABLE_LAST_BLOCK:
            trainable = True

        conv_layers = sum([True for layer in sub_model.layers if 'conv' in layer.name])
        conv_layer = 0

        for i_in, layer in enumerate(sub_model.layers):
            # Only the convolutional layers have weights to load
            if not layer.name.startswith('conv2d'):
                continue
            conv_layer += 1
            if 'last_layers' in block and trainable_option == constants.TRAINABLE_LAST_CONV \
                    and conv_layer == conv_layers:
                # If this is the last convolutional and it has ben set to trainable, set it as trainable.
                trainable = True
            if not load_last_conv and conv_layer == conv_layers and 'last_layers' in block:
                # When we don't load the last conv layer we still have to read those weights from the file,
                # otherwise, they will be assigned to another layer!
                if i_in + 1 < len(sub_model.layers) and sub_model.layers[i_in + 1].name.startswith('batch_norm'):
                    np.fromfile(wf, dtype=np.float32, count=4 * 255)
                else:
                    np.fromfile(wf, dtype=np.float32, count=255)

                np.fromfile(wf, dtype=np.float32, count=np.product((255, layer.input_shape[-1], 1, 1)))
                continue
            load_weight(i_in, layer, sub_model, wf, trainable)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()
    if save_weights_path:
        model.save_weights(save_weights_path)


def load_weight(pos, layer, sub_model, wf, trainable):
    layer_name = layer.name
    batch_norm = None
    if pos + 1 < len(sub_model.layers) and \
            sub_model.layers[pos + 1].name.startswith('batch_norm'):
        batch_norm = sub_model.layers[pos + 1]
    logging.info("{}/{} {}. {}".format(
        sub_model.name, layer.name, 'bn' if batch_norm else 'bias', "Trainable" if trainable else "Frozen"))
    filters = layer.filters
    size = layer.kernel_size[0]
    in_dim = layer.input_shape[-1]
    if batch_norm is None:
        conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
    else:
        # darknet [beta, gamma, mean, variance]
        bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
        # tf [gamma, beta, mean, variance]
        bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
    # darknet shape (out_dim, in_dim, height, width)
    conv_shape = (filters, in_dim, size, size)
    conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
    # tf shape (height, width, in_dim, out_dim)
    conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])
    if batch_norm is None:
        layer.set_weights([conv_weights, conv_bias])
    else:
        layer.set_weights([conv_weights])
        batch_norm.set_weights(bn_weights)
    layer.trainable = trainable
