import unittest

import tensorflow as tf

import constants
from data.dataset import Dataset
from models.detection.yolov3 import YOLOv3


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(constants.BDD100K)

    def test_trainable_all(self):
        def helper(model):
            model.load_for_transfer_learning(dataset=self.dataset, trainable_option=constants.TRAINABLE_ALL,
                                             for_training=False, save_weights=False)
            for block in model.train_model.layers[1:]:
                sub_model = model.train_model.get_layer(block.name)
                for layer in sub_model.layers:
                    if 'conv' not in layer.name:
                        continue
                    self.assertEqual(layer.trainable, True)

        helper(YOLOv3(tiny=False))
        helper(YOLOv3(tiny=True))

    def test_trainable_features(self):
        def helper(model):
            model.load_for_transfer_learning(dataset=self.dataset, trainable_option=constants.TRAINABLE_FEATURES,
                                             for_training=False, save_weights=False)
            for block in model.train_model.layers[1:]:
                sub_model = model.train_model.get_layer(block.name)
                for layer in sub_model.layers:
                    if 'conv' not in layer.name:
                        continue
                    if 'DarkNet' in block.name:
                        self.assertEqual(layer.trainable, False)
                    else:
                        self.assertEqual(layer.trainable, True)

        helper(YOLOv3(tiny=False))
        helper(YOLOv3(tiny=True))

    def test_trainable_last_block(self):
        def helper(model):
            model.load_for_transfer_learning(dataset=self.dataset, trainable_option=constants.TRAINABLE_LAST_BLOCK,
                                             for_training=False, save_weights=False)
            for block in model.train_model.layers[1:]:
                sub_model = model.train_model.get_layer(block.name)
                for layer in sub_model.layers:
                    if 'conv' not in layer.name:
                        continue
                    if 'last_layers' in block.name:
                        self.assertEqual(layer.trainable, True)
                    else:
                        self.assertEqual(layer.trainable, False)

        helper(YOLOv3(tiny=False))
        helper(YOLOv3(tiny=True))

    def test_trainable_last_conv(self):
        def helper(model):
            model.load_for_transfer_learning(dataset=self.dataset, trainable_option=constants.TRAINABLE_LAST_CONV,
                                             for_training=False)
            for block in model.train_model.layers[1:]:
                sub_model = model.train_model.get_layer(block.name)
                conv_layers = sum([True for layer in sub_model.layers if 'conv' in layer.name])
                conv_layer = 0
                for layer in sub_model.layers:
                    if 'conv' not in layer.name:
                        continue
                    conv_layer += 1
                    if 'last_layers' in block.name and conv_layer == conv_layers:
                        self.assertEqual(layer.trainable, True)
                    else:
                        self.assertEqual(layer.trainable, False)

        helper(YOLOv3(tiny=False))
        helper(YOLOv3(tiny=True))

    def test_transfer_learning(self):
        def helper(original_model, test_model, tiny):
            model_blocks = [l.name for l in original_model.layers[1:]]
            if tiny:
                model_blocks = ['Tiny_DarkNet', 'tiny_layer', 'last_layers_512', 'Concatenate_128', 'last_layers_256']

            for i, block in enumerate(model_blocks):
                sub_original_model = original_model.get_layer(block)
                sub_test_model = test_model.get_layer(block)
                if type(sub_original_model) != tf.keras.Model:
                    continue
                conv_layers = sum([True for layer in sub_original_model.layers if 'conv' in layer.name])
                conv_layer = 0

                for i_in, layer in enumerate(sub_original_model.layers):
                    weights_equal = True
                    if 'input' in layer.name:
                        continue
                    if layer.name.startswith('conv2d'):
                        conv_layer += 1
                        if conv_layer == conv_layers and 'last_layers' in block:
                            # This is one of the last convolutional layer. Therefore the weights should be different
                            # to the original ones
                            weights_equal = False

                    for n_weights, weights in enumerate(layer.weights):
                        original_weights = weights.numpy().flatten()
                        test_weights = sub_test_model.layers[i_in].weights[n_weights].numpy().flatten()
                        if weights_equal:
                            self.assertEqual(sum(original_weights != test_weights), 0)
                        else:
                            self.assertNotEqual(sum(original_weights[:len(test_weights)] != test_weights), 0)

        original_tiny_model = YOLOv3(tiny=True)
        original_tiny_model.load_original_yolov3(for_training=False)
        original_full_model = YOLOv3(tiny=False)
        original_full_model.load_original_yolov3(for_training=False)

        tiny_test_model = YOLOv3(tiny=True)
        full_test_model = YOLOv3(tiny=False)
        trainable_options = [constants.TRAINABLE_ALL, constants.TRAINABLE_FEATURES, constants.TRAINABLE_LAST_CONV,
                             constants.TRAINABLE_LAST_BLOCK]
        for trainable_option in trainable_options:
            tiny_test_model.load_for_transfer_learning(dataset=self.dataset, trainable_option=trainable_option,
                                                       for_training=False, save_weights=False)
            full_test_model.load_for_transfer_learning(dataset=self.dataset, trainable_option=trainable_option,
                                                       for_training=False, save_weights=False)

            helper(original_tiny_model.train_model, tiny_test_model.train_model, True)
            helper(original_full_model.train_model, full_test_model.train_model, False)


if __name__ == '__main__':
    unittest.main()
