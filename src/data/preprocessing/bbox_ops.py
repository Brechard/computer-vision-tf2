import tensorflow as tf


def check_bboxes(x_min, y_min, x_max, y_max, label):
    """ Calls the methods clip_bbox and check_bbox_area with their coordinates order """
    y_min, x_min, y_max, x_max = clip_bbox(y_min, x_min, y_max, x_max)
    y_min, x_min, y_max, x_max = check_bbox_area(y_min, x_min, y_max, x_max)
    return tf.convert_to_tensor([x_min, y_min, x_max, y_max, label])


def check_bbox_area(min_x, min_y, max_x, max_y, delta=0.05):
    """
    Adjusts bbox coordinates to make sure the area is > 0.

    :param min_y: Normalized bbox coordinate of type float between 0 and 1.
    :param min_x: Normalized bbox coordinate of type float between 0 and 1.
    :param max_y: Normalized bbox coordinate of type float between 0 and 1.
    :param max_x: Normalized bbox coordinate of type float between 0 and 1.
    :param delta: Float, this is used to create a gap of size 2 * delta between
    bbox min/max coordinates that are the same on the boundary.
    This prevents the bbox from having an area of zero.

    :returns: Tuple of new bbox coordinates between 0 and 1 that will now have a guaranteed area > 0.
    """
    height = max_y - min_y
    width = max_x - min_x

    def _adjust_bbox_boundaries(min_coord, max_coord):
        # Make sure max is never 0 and min is never 1.
        max_coord = tf.maximum(max_coord, 0.0 + delta)
        min_coord = tf.minimum(min_coord, 1.0 - delta)
        return min_coord, max_coord

    min_y, max_y = tf.cond(tf.equal(height, 0.0),
                           lambda: _adjust_bbox_boundaries(min_y, max_y),
                           lambda: (min_y, max_y))
    min_x, max_x = tf.cond(tf.equal(width, 0.0),
                           lambda: _adjust_bbox_boundaries(min_x, max_x),
                           lambda: (min_x, max_x))
    return min_x, min_y, max_x, max_y


def check_bbox_inside(bbox):
    """
    Checks that the bounding box is inside the picture with 5 checks:
        1. Is the axis_0 greater than 1?
        2. Is the axis_1 greater than 1?
        3. Is the axis_0 less than 0?
        4. Is the axis_1 less than 0?
        5. Are all the coordinates 0?

    :param bbox: Bounding box coordinates. The order of them is not important as longs as the
    elements 0 and 2, and 1 and 3 correspond to same axis.

    :return: Flag indicating if is inside the image.
             Input if it is inside the image, otherwise [0.0, 0.0, 0.0, 0.0, 0.0].
    """
    axis_0 = [bbox[0], bbox[2]]
    axis_1 = [bbox[1], bbox[3]]
    if tf.reduce_sum(tf.cast(tf.math.greater(axis_0, 1), tf.float32)) == 2 or \
            tf.reduce_sum(tf.cast(tf.math.greater(axis_1, 1), tf.float32)) == 2 or \
            tf.reduce_sum(tf.cast(tf.math.less(axis_0, 0), tf.float32)) == 2 or \
            tf.reduce_sum(tf.cast(tf.math.less(axis_1, 0), tf.float32)) == 2 or \
            tf.reduce_sum(tf.cast(tf.math.equal(bbox[:4], 0), tf.float32)) == 4:
        # This bounding box is outside the image
        return False, tf.convert_to_tensor([0, 0, 0, 0, 0], dtype=tf.float32)
    else:
        return True, bbox


def check_clip_bbox_fn(bbox):
    """
    If the bounding box is inside it clips the bounding box, otherwise it returns a void bounding box.

    :param bbox: 1-D float Tensor with the coordinates of the bounding box and its label.

    :return: void bounding box if it is not inside the image, otherwise returns the clipped bounding box.
    """
    inside, void_bbox = check_bbox_inside(bbox)
    if not inside:
        return void_bbox
    p0, p1, p2, p3 = clip_bbox(bbox[0], bbox[1], bbox[2], bbox[3])
    return tf.convert_to_tensor([p0, p1, p2, p3, bbox[4]])


def clip_bbox(p0, p1, p2, p3):
    """
    Clip bounding box coordinates between 0 and 1.

    :param p0: Normalized bbox coordinate of type float between 0 and 1.
    :param p1: Normalized bbox coordinate of type float between 0 and 1.
    :param p2: Normalized bbox coordinate of type float between 0 and 1.
    :param p3: Normalized bbox coordinate of type float between 0 and 1.

    :returns: Clipped coordinate values between 0 and 1.
    """
    p0 = tf.clip_by_value(p0, 0.0, 1.0)
    p1 = tf.clip_by_value(p1, 0.0, 1.0)
    p2 = tf.clip_by_value(p2, 0.0, 1.0)
    p3 = tf.clip_by_value(p3, 0.0, 1.0)
    return p0, p1, p2, p3


def denormalize_bbox(x_min, y_min, x_max, y_max, image_width, image_height):
    """ Denormalizes the coordinates of a bounding box """
    return x_min * image_width, y_min * image_height, x_max * image_width, y_max * image_height


def flip_bbox(bbox):
    """ Flip the bounding box horizontally if it is not a void box. Otherwise it returns the input """
    if tf.reduce_sum(bbox[:4]) == 0:
        return bbox
    return tf.convert_to_tensor([1 - bbox[2], bbox[1], 1 - bbox[0], bbox[3], bbox[4]])


def flip_bbox_coordinates(bbox):
    """ Flips the y and x coordinates of a bounding box without label """
    return tf.convert_to_tensor([bbox[1], bbox[0], bbox[3], bbox[2]])


def flip_bbox_coordinates_label(bbox):
    """ Flips the y and x coordinates of a bounding box with label """
    return tf.convert_to_tensor([bbox[1], bbox[0], bbox[3], bbox[2], bbox[4]])


def normalize_bbox(x_min, y_min, x_max, y_max, image_width, image_height):
    """ Normalizes the coordinates of a bounding box """
    return x_min / image_width, y_min / image_height, x_max / image_width, y_max / image_height


def rotate_bbox(bbox, image_width, image_height, radians, center):
    """
    Rotates the bounding box by radians in respect to the center passed as parameter

    :param bbox: 1-D float32 Tensor that is a bounding box in the image.
    It has 5 elements (min_y, min_x, max_y, max_x, label). Each coordinate is normalized.
    :param image_width: Width of the image.
    :param image_height: Height of the image.
    :param radians: Radians to rotate the bounding box.
    :param center: Center of the rotation.

    :return: void bounding box if it is not inside the image before or after rotation,
    otherwise the rotated bounding box.
    """
    inside, bbox = check_bbox_inside(bbox)
    if not inside:
        return bbox
    bbox, label = bbox[:-1], bbox[-1]
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    # Calculate the 4 points of the bounding box
    p1 = tf.convert_to_tensor([bbox[0], bbox[1]])
    p2 = tf.convert_to_tensor([bbox[0] + bbox_width, bbox[1]])
    p3 = tf.convert_to_tensor([bbox[2], bbox[3]])
    p4 = tf.convert_to_tensor([bbox[0], bbox[1] + bbox_height])
    # De-Normalize the 4 points of the box
    pts = tf.transpose(tf.multiply([p1, p2, p3, p4], [image_width, image_height]))
    new_bbox = rotate_points(pts, radians, center)
    # Select the new x_min, y_min, x_max, y_max in the rotated points
    x_min = tf.math.reduce_min(new_bbox[0, :])
    y_min = tf.math.reduce_min(new_bbox[1, :])
    x_max = tf.math.reduce_max(new_bbox[0, :])
    y_max = tf.math.reduce_max(new_bbox[1, :])
    x_min, y_min, x_max, y_max = normalize_bbox(x_min, y_min, x_max, y_max, image_width, image_height)
    inside, void_bbox = check_bbox_inside(tf.convert_to_tensor([x_min, y_min, x_max, y_max]))
    if not inside:
        return void_bbox

    return check_bboxes(x_min, y_min, x_max, y_max, label)


def rotate_points(pts, theta, rc):
    """
    Code from: https://scipython.com/blog/the-bounding-box-of-a-rotated-object/

    Rotate the (x,y) points pts by angle theta about centre rc.
    """
    c, s = tf.math.cos(theta), tf.math.sin(theta)
    R = tf.convert_to_tensor([[c, -s], [s, c]], dtype=tf.float32)
    return rc + R @ (pts - rc)


def shear_bbox_horizontal(bbox, image_width, image_height, level, axis):
    """
    Shear the bounding box by level in respect to the axis horizontally.

    :param bbox: 1-D float32 Tensor that is a bounding box in the image.
    It has 5 elements (min_y, min_x, max_y, max_x, label). Each coordinate is normalized.
    :param image_width: Width of the image.
    :param image_height: Height of the image.
    :param level: How much to shear the bounding box in radians.
    :param axis: Axis of shearing.

    :return: void bounding box if it is not inside the image before or after shearing,
    otherwise the sheared bounding box.
    """
    inside, bbox = check_bbox_inside(bbox)
    if not inside:
        return bbox
    # De-Normalize the 2 points of the bbox
    bbox = tf.multiply(bbox, [image_width, image_height, image_width, image_height, 1])
    # Apply the transformations, it only applies to the y axis
    y_min_delta = bbox[1] - axis
    y_max_delta = bbox[3] - axis
    # Instead of calculating the 4 points, we have a flag to know how to calculate the new points
    if level < 0:
        x_min_delta = y_min_delta * tf.math.atan(level)
        x_max_delta = y_max_delta * tf.math.atan(level)
    else:
        x_min_delta = y_max_delta * tf.math.atan(level)
        x_max_delta = y_min_delta * tf.math.atan(level)

    x_min = bbox[0] - x_min_delta
    x_max = bbox[2] - x_max_delta
    x_min, y_min, x_max, y_max = normalize_bbox(x_min, bbox[1], x_max, bbox[3], image_width, image_height)
    inside, void_bbox = check_bbox_inside(tf.convert_to_tensor([x_min, y_min, x_max, y_max]))
    if not inside:
        return void_bbox

    return check_bboxes(x_min, y_min, x_max, y_max, bbox[-1])


def shear_bbox_vertical(bbox, image_width, image_height, level, axis):
    """
    Shear the bounding box by level in respect to the axis vertically.

    :param bbox: 1-D float32 Tensor that is a bounding box in the image.
    It has 5 elements (min_y, min_x, max_y, max_x, label). Each coordinate is normalized.
    :param image_width: Width of the image.
    :param image_height: Height of the image.
    :param level: How much to shear the bounding box in radians.
    :param axis: Axis of shearing.

    :return: void bounding box if it is not inside the image before or after shearing,
    otherwise the sheared bounding box.
    """

    inside, bbox = check_bbox_inside(bbox)
    if not inside:
        return bbox

    # De-Normalize the 2 points of the bbox
    bbox = tf.multiply(bbox, [image_width, image_height, image_width, image_height, 1])
    # Apply the transformations, it only applies to the y axis
    x_min_delta = bbox[0] - axis
    x_max_delta = bbox[2] - axis
    if level < 0:
        y_min_delta = x_min_delta * tf.math.atan(level)
        y_max_delta = x_max_delta * tf.math.atan(level)
    else:
        y_min_delta = x_max_delta * tf.math.atan(level)
        y_max_delta = x_min_delta * tf.math.atan(level)
    y_min = bbox[1] - y_min_delta
    y_max = bbox[3] - y_max_delta
    x_min, y_min, x_max, y_max = normalize_bbox(bbox[0], y_min, bbox[2], y_max, image_width, image_height)
    inside, void_bbox = check_bbox_inside(tf.convert_to_tensor([x_min, y_min, x_max, y_max]))
    if not inside:
        return void_bbox

    return check_bboxes(x_min, y_min, x_max, y_max, bbox[-1])
