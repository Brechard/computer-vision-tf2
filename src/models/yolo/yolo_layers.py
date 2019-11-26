import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2


def concat_layers(n_filters: int, upsample_features: tf.Tensor,
                  skip_features: tf.Tensor) -> tf.keras.layers.Concatenate:
    """ Layers to concatenate the features from bigger features that have to be upsampled and the features
        from the scale we are detecting in this level """
    x, x_skip = inputs = layers.Input(upsample_features.shape[1:]), layers.Input(skip_features.shape[1:])
    x = conv_2d(x, n_filters, 1, 1)
    x = layers.UpSampling2D(2)(x)
    x = layers.Concatenate()([x, x_skip])
    return tf.keras.Model(inputs, x, name="Concatenate_" + str(n_filters))((upsample_features, skip_features))


def conv_block(x, n_filters_list: list, kernel_size_list: list, n_blocks: int, strides_list: list = None):
    """ Convolutional block, the features to reuse are not the output from the last convolutional layer
        but the previous one """
    assert len(n_filters_list) == len(kernel_size_list)

    for block in range(n_blocks):
        for conv in range(len(n_filters_list)):
            stride = (1 if strides_list is None else strides_list[conv])
            x = conv_2d(x, n_filters=n_filters_list[conv], kernel_size=kernel_size_list[conv], stride=stride)
            if block == n_blocks - 1 and conv == len(n_filters_list) - 2:
                x_features = x
    return x, x_features


def conv_2d(x, n_filters: int, kernel_size: int, stride: int, max_pool: bool = False):
    """
    Conv2D layer
    :param x: input to the layer
    :param n_filters: Number of filters for the layer
    :param kernel_size: Size of the kernel
    :param stride: Stride to apply
    :param max_pool: Add a max pooling layer at the end, used in tiny yolo
    :return: layer
    """
    if stride == 1:
        padding = 'same'
    else:
        x = layers.ZeroPadding2D(((1, 0), (1, 0)))(x)  # top left half-padding
        padding = 'valid'
    # Using bias and Batch normalization makes no sense, therefore we turn it off
    x = layers.Conv2D(n_filters, kernel_size, stride, padding, kernel_regularizer=l2(0.0005), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    if max_pool:
        x = layers.MaxPool2D(2, 2, 'same')(x)
    return x


def darknet_residual_block(x, n_filters_list: list, kernel_sizes_list: list, n_blocks: int, strides_list: list = None):
    """
    Darknet residual block consist of a chain of n residual blocks all the same
    :param x: input of the block
    :param n_filters_list: List with the number of filters for each conv layer
    :param kernel_sizes_list: List with the sizes of the kernels for each conv layer
    :param n_blocks: Number of blocks
    :param strides_list: list with the stride for each conv layer
    :return: output of the last residual block
    """
    for _ in range(n_blocks):
        x = residual_block(x, n_filters_list, kernel_sizes_list, strides_list)
    return x


def last_layers(last_layers_input, n_filters_list, kernel_size_list, n_anchors, n_classes):
    """ The last layers of the three scales are the same. A convolutional block that extracts features and a
        convolutional layer. The last lambda layer reshapes the output so that each anchor mask prediction is
        in a different row """
    x = input = layers.Input(last_layers_input.shape[1:])
    x, x_features = conv_block(x, n_filters_list=n_filters_list, kernel_size_list=kernel_size_list, n_blocks=3)
    x = layers.Conv2D(n_anchors * (n_classes + 5), 1, 1, kernel_regularizer=l2(0.0005), use_bias=True)(x)
    x = layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], n_anchors, n_classes + 5)))(x)
    return tf.keras.Model(input, (x, x_features), name="last_layers_" + str(n_filters_list[0]))(last_layers_input)


def last_tiny_layers(last_layers_input, n_filters, kernel_size, n_anchors, n_classes):
    """ The last layers of tiny_yolo_v3 do not have convolutional blocks.
        The last lambda layer reshapes the output so that each anchor mask prediction is
        in a different row """
    x = input = layers.Input(last_layers_input.shape[1:])
    x = conv_2d(x, n_filters, kernel_size, 1)
    x = layers.Conv2D(n_anchors * (n_classes + 5), 1, 1, kernel_regularizer=l2(0.0005), use_bias=True)(x)
    x = layers.Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], n_anchors, n_classes + 5)))(x)
    return tf.keras.Model(input, x, name="last_layers_" + str(n_filters))(last_layers_input)


def tiny_layer(x_big_features):
    x = input = layers.Input(x_big_features.shape[1:])
    x = conv_2d(x, n_filters=256, kernel_size=1, stride=1)
    return tf.keras.Model(input, x, name='tiny_layer')(x_big_features)


def residual_block(x, n_filters_list: list, kernel_sizes_list: list, strides_list: list = None):
    """
    Residual block of convolutional layers
    :param x: input of the block
    :param n_filters_list: List with the number of filters for each conv layer
    :param kernel_sizes_list: List with the sizes of the kernels for each conv layer
    :param strides_list: list with the stride for each conv layer
    :return: output of the residual block
    """
    assert len(n_filters_list) == len(kernel_sizes_list)
    original_input = x
    for conv_layer in range(len(n_filters_list)):
        stride = (1 if strides_list is None else strides_list[conv_layer])
        x = conv_2d(x, n_filters_list[conv_layer], kernel_sizes_list[conv_layer], stride)
    x = layers.Add()([original_input, x])

    return x
