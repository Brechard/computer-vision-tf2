import collections
import os
import pickle
from random import sample

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import constants
import helpers
from models.recognition.recognizer import Recognizer
from visualization.visualize import plot_3x3_images_RW


def analyze(checkpoint_dir_path: str, dataset_name: str = constants.GTSR, model_img_res: int = 30, plot: bool = False,
            n_plots: int = 10, verbose: int = 0):
    with open(checkpoint_dir_path + 'neuron_to_class_dict.p', "rb") as f:
        neuron_to_class_dict = pickle.load(f)

    model = Recognizer(dataset_name, model_img_res, False, 'mini')
    model.inference_model.load_weights(checkpoint_dir_path + 'weights.ckpt')
    images_path = constants.DATASET_PATH.format(dataset_name) + 'test/'

    test = pd.read_csv(constants.DATASET_PATH.format(dataset_name) + 'Test.csv')
    test['Path'] = test['Path'].apply(lambda x: x.split('/')[-1])
    test = test.set_index('Path')
    classified = {}
    n_wrong, total = 0, 0
    for i in test['ClassId'].unique():
        classified[str(i)] = collections.defaultdict(int)

    image_paths, titles, n_ploted = [], [], 0
    images_paths = os.listdir(images_path)
    for i, image_id in enumerate(tqdm(sample(images_paths, len(images_paths)))):
        if 'png' not in image_id:
            continue
        total += 1
        image_path = images_path + image_id
        img = tf.image.decode_image(open(image_path, 'rb').read(), channels=3)
        img = tf.expand_dims(helpers.transform_images(img, model.img_res), 0)
        real_class_id = str(test.loc[image_id]['ClassId'])

        img_classes_probs = model.inference_model(img)
        img_class_id = np.argmax(img_classes_probs)
        img_class_id = neuron_to_class_dict[str(img_class_id)]
        classified[real_class_id][img_class_id] += 1

        if real_class_id != img_class_id:
            n_wrong += 1
        if verbose > 0:
            img_class_prob = np.max(img_classes_probs.numpy())
            if real_class_id != img_class_id:
                print(constants.C_OKBLUE, "Correctly labeled as", model.labels_map_dict[str(img_class_id)].upper(),
                      'with probability {:.2f} %'.format(100 * img_class_prob), constants.C_ENDC)
            else:
                print(constants.C_FAIL, "Wrongly labeled as", model.labels_map_dict[str(img_class_id)].upper(),
                      ', id =',
                      img_class_id, 'with probability {:.2f} %'.format(100 * img_class_prob), '. Should have been',
                      model.labels_map_dict[real_class_id],
                      ', id =', real_class_id, constants.C_ENDC)
        if plot:
            if i % 9 == 0 and i > 0 and n_ploted < n_plots:
                plot_3x3_images_RW(image_paths, titles)
                image_paths, titles = [], []
                n_ploted += 1
            image_paths.append(image_path)
            pred_class = ("R" if real_class_id == img_class_id else "W") + model.labels_map_dict[str(img_class_id)]
            titles.append(pred_class.upper())

    error_rate = 100 * n_wrong / total
    for real_class, value in classified.items():
        print('Class', real_class)
        for pred_class, times in value.items():
            if pred_class != real_class:
                print("Misslabeled for class", pred_class, '->', times, 'times')
            else:
                print("Correctly labeled", times, 'times')
        print()

    print("Error rate (% of missclassified) = {:.2f} % in test.".format(error_rate))
    miss_class_perc = {}
    for real_class, predictions in classified.items():
        total = 0
        wrong = 0
        for pred_class, times in predictions.items():
            total += times
            if real_class != pred_class:
                wrong += times
            miss_class_perc[real_class] = 100 * wrong / total

    def plot_helper(show_labels):
        fig, ax = plt.subplots(1, 1)
        ticks = []
        for real_class, class_error_rate in miss_class_perc.items():
            if show_labels:
                ticks.append(model.labels_map_dict[str(real_class)] if class_error_rate > error_rate else '')
            else:
                ticks.append(str(real_class) if class_error_rate > error_rate else '')

        ax.bar(range(len(ticks)), miss_class_perc.values())
        ax.plot([-1, len(ticks)], [error_rate, error_rate], 'k--')
        ax.set_xticks([i for i in range(len(miss_class_perc))])
        if show_labels:
            ax.set_xticklabels(ticks, rotation='vertical')
        else:
            ax.set_xticklabels(ticks, rotation='horizontal')
            plt.xlabel('Classes')

        plt.ylabel('Percentage of miss-classifications')
        plt.title('Miss-classification percentages per label')
        plt.xlim([-1, len(ticks)])
        plt.tight_layout()
        plt.show()

    plot_helper(True)
    plot_helper(False)

    return classified, error_rate
