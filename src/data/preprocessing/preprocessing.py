from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow_addons as tfa

from data.preprocessing import box_list
from data.preprocessing import box_list_ops
from data.preprocessing.bbox_ops import *


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """
    Base code can be found in:
    https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py

    Distort the color of a Tensor image.

    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.

    :param image: 3-D Tensor containing single image in [0, 1].
    :param color_ordering: Python int, a type of distortion (valid values: 0-4).
    :param fast_mode: Avoids slower ops (random_hue and random_contrast)
    :param scope: Optional scope for name_scope.
    :returns: 3-D Tensor color-distorted image on range [0, 1]

    """
    lower = 0.5
    upper = 1.5
    with tf.name_scope('distort_color' if scope is None else scope):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=lower, upper=upper)
            else:
                image = tf.image.random_saturation(image, lower=lower, upper=upper)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=lower, upper=upper)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=lower, upper=upper)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=lower, upper=upper)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=lower, upper=upper)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=lower, upper=upper)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=lower, upper=upper)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=lower, upper=upper)
                image = tf.image.random_contrast(image, lower=lower, upper=upper)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            elif color_ordering == 4:
                # Fast mode of color color_ordering
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=lower, upper=upper)
            elif color_ordering == 5:
                # Fast mode v2 of color color_ordering
                image = tf.image.random_saturation(image, lower=lower, upper=upper)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            elif color_ordering == 6:
                # Fast mode v2 of only changing the colors
                image = tf.image.random_hue(image, max_delta=0.2)
            else:
                # Don't do any transformations
                return image

        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def distorted_bounding_box_crop(image, bbox, min_object_covered=0.4, aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.3, 1.0), max_attempts=10, scope=None):
    """
    Base code can be found in:
    https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py

    Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    :param image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    :param bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
    where each coordinate is [0, 1) and the coordinates are arranged as [ymin, xmin, ymax, xmax].
    If num_boxes is 0 then it would use the whole image.
    :param min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
    area of the image must contain at least this fraction of any bounding box supplied.
    :param aspect_ratio_range: An optional list of `floats`. The cropped area of the
    image must have an aspect ratio = width / height within this range.
    :param area_range: An optional list of `floats`. The cropped area of the image
    must contain a fraction of the supplied image within in this range.
    :param max_attempts: An optional `int`. Number of attempts at generating a cropped
    region of the image of the specified constraints. After `max_attempts` failures, return the entire image.
    :param scope: Optional scope for name_scope.

    :returns: A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope('distorted_bounding_box_crop' if scope is None else scope):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox


def preprocess_for_train(image, height, width, bboxes, fast_mode=False, scope=None):
    """
    Base code can be found in:
    https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py

    Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    :param image: 3-D Tensor [height, width, channels] with the image. If dtype is tf.float32 then
    the range should be [0, 1], otherwise it would converted to tf.float32 assuming that
    the range is [0, MAX], where MAX is largest positive representable number for int(8/16/32)
    data type (see `tf.image.convert_image_dtype` for details).
    :param height: integer, output image height.
    :param width: integer, output image width.
    :param bboxes: 2-D float32 Tensor that is a list of the bounding boxes in the image.
    Each bbox has 5 elements (min_y, min_x, max_y, max_x, label). Each coordinate is normalized.
    :param fast_mode: Optional boolean, if True avoids slower transformations
    (i.e. bi-cubic resizing, random_hue or random_contrast).
    :param scope: Optional scope for name_scope.

    :returns: 3-D float Tensor of distorted image used for training with range [-1, 1].
              2-D float Tensor of bounding boxes after the image transformation
    """
    # image_idx = 0
    with tf.name_scope('distort_image' if scope is None else scope):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # image_idx = pib(image, bboxes, title="Original Image", switch=False, dataset_name="COCO", i=image_idx)
        image = tf.image.random_jpeg_quality(image, 0, 100)
        # image_idx = pib(image, bboxes, title="Random JPEG quality", switch=False, dataset_name="COCO", i=image_idx)
        bboxes, image, _ = transform_image(bboxes, image)
        # image_idx = pib(image, bboxes, title="Image transformed", switch=False, dataset_name="COCO", i=image_idx)

        new_bboxes = tf.map_fn(lambda bbox: flip_bbox_coordinates(bbox), bboxes)
        distorted_image, distorted_bbox = distorted_bounding_box_crop(image, [new_bboxes])
        # Restore the shape since the dynamic slice based upon the bbox_size loses
        # the third dimension.
        # dbbox = distorted_bbox[0].numpy()
        distorted_image.set_shape([None, None, 3])
        image_with_distorted_box = tf.image.draw_bounding_boxes(
            tf.expand_dims(image, 0), distorted_bbox, None)

        # image_idx = pib(image_with_distorted_box[0], tf.concat([new_bboxes, bboxes[:, -1:]], axis=1), switch=True,
        #                 title="Transformed image with crop", dataset_name="COCO", i=image_idx)

        boxlist = box_list.BoxList(tf.convert_to_tensor(new_bboxes))
        im_box_rank1 = tf.squeeze(distorted_bbox)
        new_bboxes = box_list_ops.change_coordinate_frame(boxlist, im_box_rank1).get()
        bboxes = tf.concat([new_bboxes, bboxes[:, -1:]], axis=1)
        bboxes = tf.map_fn(lambda bbox: check_clip_bbox_fn(bbox), bboxes)
        bboxes = tf.map_fn(lambda bbox: flip_bbox_coordinates_label(bbox), bboxes)

    distorted_image = tf.image.resize(distorted_image, [height, width])
    # image_idx = pib(distorted_image, bboxes, title="Cropped image with fix size", dataset_name="COCO",
    #                 i=image_idx)

    old_distorted_image = distorted_image
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(old_distorted_image)

    if tf.reduce_sum(tf.cast(False == (old_distorted_image == distorted_image), tf.float32)) != 0:
        # The image has been flipped
        bboxes = tf.map_fn(lambda bbox: flip_bbox(bbox), bboxes)
        # image_idx = pib(distorted_image, bboxes, title="Flipped image", i=image_idx, dataset_name="COCO")

    # Randomly distort the colors. This is the actual bottleneck!
    ordering = random.randint(0, 10)
    distorted_image = distort_color(distorted_image, ordering, fast_mode)

    # pib(distorted_image, bboxes, title="Final Image", dataset_name='COCO', i=image_idx)
    # distorted_image = tf.subtract(distorted_image, 0.5)
    # distorted_image = tf.multiply(distorted_image, 2.0)
    return distorted_image, tf.convert_to_tensor(bboxes)


def preprocess_for_eval(image, height, width, bboxes, central_fraction=0.875, scope=None, central_crop=True):
    """
    Base code can be found in:
    https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py

    Prepare one image for evaluation.

    If height and width are specified it would output an image with that size by
    applying resize_bilinear.

    If central_fraction is specified it would crop the central fraction of the
    input image.

    :param image: 3-D Tensor [height, width, channels] with the image. If dtype is tf.float32 then
    the range should be [0, 1], otherwise it would converted to tf.float32 assuming that
    the range is [0, MAX], where MAX is largest positive representable number for int(8/16/32)
    data type (see `tf.image.convert_image_dtype` for details).
    :param height: integer, output image height.
    :param width: integer, output image width.
    :param bboxes: 2-D float32 Tensor that is a list of the bounding boxes in the image.
    Each bbox has 5 elements (min_y, min_x, max_y, max_x, label). Each coordinate is normalized.
    :param central_fraction: Optional Float, fraction of the image to crop.
    :param scope: Optional scope for name_scope.
    :param central_crop: Enable central cropping of images during preprocessing for evaluation.

    :returns: 3-D float Tensor of image.
              2-D float Tensor of bounding boxes (input).
    """
    with tf.name_scope('eval_image' if scope is None else scope):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        if central_crop and central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)

        if height and width:
            # Resize the image to the specified height and width.
            image = tf.expand_dims(image, 0)
            image = tf.compat.v1.image.resize_bilinear(image, [height, width],
                                                       align_corners=False)
            image = tf.squeeze(image, [0])
        # image = tf.subtract(image, 0.5)
        # image = tf.multiply(image, 2.0)

        return image, bboxes


def preprocess_image(image, output_height, output_width, bboxes, is_training=False, fast_mode=False):
    """
    Base code can be found in:
    https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py

    Pre-process one image for training or evaluation.
    
    :param image: 3-D Tensor [height, width, channels] with the image. If dtype is tf.float32 then 
    the range should be [0, 1], otherwise it would converted to tf.float32 assuming that 
    the range is [0, MAX], where MAX is largest positive representable number for int(8/16/32) 
    data type (see `tf.image.convert_image_dtype` for details).
    :param output_height: integer, output image height.
    :param output_width: integer, output image width.
    :param bboxes: 2-D float32 Tensor that is a list of the bounding boxes in the image. 
    Each bbox has 5 elements (min_y, min_x, max_y, max_x, label). Each coordinate is normalized.
    :param is_training: Boolean. If true it would transform an image for train,
        otherwise it would transform it for evaluation.
    :param fast_mode: Optional boolean, if True avoids slower transformations.

    :returns: 3-D float Tensor of prepared image.
              2-D float Tensor of bounding boxes (input).
    """
    if is_training:
        return preprocess_for_train(
            image,
            output_height,
            output_width,
            bboxes,
            fast_mode)
    else:
        return preprocess_for_eval(image, output_height, output_width, bboxes=bboxes)


def rotate_with_bboxes(image, bboxes, radians):
    """
    Applies rotation transformation to the image and the bounding boxes

    :param image: 3-D float32 Tensor.
    :param bboxes: 2-D float32 Tensor that is a list of the bounding boxes in the image.
    Each bbox has 5 elements (min_y, min_x, max_y, max_x, label). Each coordinate is normalized.
    :param radians: Float, a scalar angle in degrees to rotate all images by.

    :returns: 3-D float32 Tensor, result of rotating image by degrees.
              2-D float32 Tensor, bounding boxes, shifted to reflect the rotated image.
    """
    original_input = image
    image_height = tf.cast(tf.shape(image)[0], tf.float32)
    image_width = tf.cast(tf.shape(image)[1], tf.float32)

    # Rotate the image.
    image = tfa.image.rotate(image, -radians)
    center = tf.transpose([[image_width / 2, image_height / 2]])
    radians = tf.convert_to_tensor(radians, dtype=tf.float32)
    new_bboxes = tf.map_fn(lambda bbox: rotate_bbox(bbox, image_width, image_height, radians, center), bboxes)

    return image, new_bboxes


def shear_with_bboxes(image, bboxes, level):
    """
    Applies Shear Transformation to the image and shifts the bboxes.

    :param image: 3-D float32 Tensor.
    :param bboxes: 2-D float32 Tensor that is a list of the bounding boxes in the image.
    Each bbox has 5 elements (min_y, min_x, max_y, max_x, label). Each coordinate is normalized.
    :param level: Float. How much to shear the image in radians.

    :returns: 3-D float32 Tensor, result of shearing image by level.
              2-D float32 Tensor, bounding boxes, shifted to reflect the sheared image.
    """
    image_height = tf.cast(tf.shape(image)[0], tf.float32)
    image_width = tf.cast(tf.shape(image)[1], tf.float32)
    if np.random.rand() > 0.5:
        axis = tf.random.uniform([1], 0, image_width)[0]
        transformations = [1.0, 0.0, 0.0,
                           level, 1.0, - axis * level,
                           0, 0]  # 1
        new_bboxes = tf.map_fn(lambda bbox: shear_bbox_vertical(bbox, image_width, image_height, level, axis), bboxes)
    else:
        axis = tf.random.uniform([1], 0, image_height)[0]
        transformations = [1.0, level, - axis * level,
                           0, 1.0, 0,
                           0, 0]  # 1
        new_bboxes = tf.map_fn(lambda bbox: shear_bbox_horizontal(bbox, image_width, image_height, level, axis), bboxes)

    image = tfa.image.transform(image, transformations)
    return image, new_bboxes


def transform_image(bboxes, distorted_image, image_idx=None):
    # image_idx = pib(distorted_image, bboxes, title="Cropped Image. Filtered bboxes", i=image_idx, dataset_name="COCO")
    option = tf.random.uniform([1], 0, 8, tf.int32)[0]
    # Half of the times there will be no transformation
    if option == 0:
        radians = tf.random.uniform([1], 0, 0.3)[0]
        distorted_image, bboxes = rotate_with_bboxes(distorted_image, bboxes, radians)
        # print('Option 0. radians', radians)
        # image_idx = pib(distorted_image, bboxes, title="Rotated Image", i=image_idx, dataset_name="COCO")
    elif option == 1:
        level = tf.random.uniform([1], 0, 0.3)[0]
        distorted_image, bboxes = shear_with_bboxes(distorted_image, bboxes, level)
        # print('Option 1. level', level)
        # image_idx = pib(distorted_image, bboxes, title="Shear Image", i=image_idx, dataset_name="COCO")
    elif option == 2:
        radians = tf.random.uniform([1], 0, 0.3)[0]
        distorted_image, bboxes = rotate_with_bboxes(distorted_image, bboxes, radians)
        level = tf.random.uniform([1], 0, 0.3)[0]
        distorted_image, bboxes = shear_with_bboxes(distorted_image, bboxes, level)
        # print("Option 2. radians:", radians, 'level', level)
        # image_idx = pib(distorted_image, bboxes, title="Rotate and Shear Image", i=image_idx, dataset_name="COCO")
    elif option == 3:
        level = tf.random.uniform([1], 0, 0.3)[0]
        distorted_image, bboxes = shear_with_bboxes(distorted_image, bboxes, level)
        radians = tf.random.uniform([1], 0, 0.3)[0]
        distorted_image, bboxes = rotate_with_bboxes(distorted_image, bboxes, radians)
        # print('Option 3. level', level, 'radians', radians)
        # image_idx = pib(distorted_image, bboxes, title="Shear and Rotate Image", i=image_idx, dataset_name="COCO")
    return bboxes, distorted_image, image_idx
