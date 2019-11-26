import pickle
from random import sample

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import constants
import helpers


def get_colors(annotations: list, labels: list) -> list:
    """ Find the number of unique labels and assign each of them a color from the predefined set randomly """
    for bbox in annotations:
        x_min, y_min, x_max, y_max, label = bbox
        if label not in labels:
            labels.append(label)
    colors = np.array(sample(constants.colors, len(labels))) / 255
    return colors


def histogram_labels(dataset_name: str, double_bins: bool = False, show_labels: bool = False):
    """ Plot an histogram of the labels. double_bins allow to have have more space between bins in the plot """
    annotations_df = pd.read_csv(constants.ANNOTATIONS_TRAIN_CSV_PATH.format(dataset_name))
    # if use_label_name:
    labels_values = list(helpers.get_labels_dict(dataset_name).values())
    labels_ids = annotations_df['label'].unique()
    n_bins = len(labels_ids)
    if double_bins:
        n_bins *= 2
    fig, ax = plt.subplots()
    bins = np.arange(n_bins + 1) - 0.5
    counts, bins, patches = ax.hist(annotations_df['label'], bins=bins)
    plt.xlim([-1, n_bins])
    if show_labels:
        plt.xticks(range(n_bins))

        # Label below the x-axis...
        bin_centers = bins + 0.5
        i = 0
        for count, x in zip(counts, bin_centers):
            ax.annotate(labels_values[i], (x, 0), xycoords=('data', 'axes fraction'),
                        xytext=(0, -18), textcoords='offset points', va='top', ha='center', rotation=-90)
            i += 1
        plt.subplots_adjust(bottom=0.15)
    plt.show()


def plot_3x3_images_RW(image_paths: list, titles: list):
    """ Plots 3x3 images with their title. The first letter of each image is either a W (Wrong classification) or a
        R (Right classification). """
    for i, image_path in enumerate(image_paths):
        plt.subplot(3, 3, i + 1)
        # Display the image
        plt.imshow(mpimg.imread(image_path))
        title_obj = plt.title(titles[i][1:])
        if 'W' in titles[i][0]:
            plt.setp(title_obj, color='r')
        else:
            plt.setp(title_obj, color='g')
        plt.axis('off')
    plt.show()


def plot_history(history: dict, figs_path: str = None):
    n_epochs = len(history['loss'])
    epochs = [i + 1 for i in range(n_epochs)]
    plt.plot(epochs, history['loss'], label='Train loss')
    plt.plot(epochs, history['val_loss'], label='Validation loss')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    if figs_path:
        plt.savefig(figs_path + 'loss.png')
    plt.show()

    if 'accuracy' in history:
        plt.figure()
        plt.plot(epochs, history['accuracy'], label='Train accuracy')
        plt.plot(epochs, history['val_accuracy'], label='Validation accuracy')
        plt.title('Accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend()
        if figs_path:
            plt.savefig(figs_path + 'accuracy.png')
        plt.show()

    if 'learning_rates' in history:
        plt.figure()
        plt.plot(epochs, history['learning_rates'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning rate')
        plt.title('Learning rate cosine decay')
        plt.legend()
        if figs_path:
            plt.savefig(figs_path + 'lr_schedule.png')
        plt.show()


def plot_history_from_pickle_path(path):
    history = pickle.load(open(path, "rb"))
    plot_history(history)


def plot_image_annotations(image_path: str, annotations: list, labels_map_dict: dict, probs: list = None,
                           title: str = "", output_path: str = None):
    """
    Plot one image with its bounding boxes
    :param image_path: path to the image
    :param annotations: list with the annotations [[x_min, y_min, x_max, y_max, label], [...], ...]
    :param labels_map_dict: dictionary with key the label id and value the label value
    :param probs: list with the probability of each label
    :param title: title of the plot
    :param output_path: path to save the figure. If none then it is not saved.
    """
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(mpimg.imread(image_path))
    labels = []

    # Get the unique labels to assign a color to each
    colors = get_colors(annotations, labels)

    for i, bbox in enumerate(annotations):
        plot_image_annotations_help(ax, bbox, colors, labels, labels_map_dict, False,
                                    probability=(None if probs is None else probs[i]))
    plt.title(title)
    if output_path:
        plt.savefig(output_path)
    plt.show()


def plot_images_annotations_box(images_path: list, annotations: list, labels_map: dict, title: str = ""):
    """ This code only plots 4 images in a 2x2 set. Tailor it to your needs! """
    if len(images_path) != len(annotations) != 4:
        raise Exception("Each image needs to have their annotations!")

    # Create figure and axes
    fig, ax = plt.subplots(2, 2)
    i = 0
    for index, path in enumerate(images_path):
        j = index % 2
        if index == 2:
            i = 1
        # Display the image
        ax[i, j].imshow(mpimg.imread(path))
        labels = []
        colors = get_colors(annotations[index], labels)

        # Plot the bounding boxes and labels
        for bbox in annotations[index]:
            plot_image_annotations_help(ax[i, j], bbox, colors, labels, labels_map, False)
        ax[i, j].set_title(path.split('/')[-1], fontsize=7)

    plt.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(title)
    plt.show()


def plot_image_annotations_help(ax, bbox, colors, labels, labels_map, axis, probability=None):
    """ Plot the image and its bounding box """
    x_min, y_min, x_max, y_max, label = bbox
    c = colors[labels.index(label)]
    # if label == 43:
    #     a = 1
    label = labels_map[str(label)]
    if probability is not None:
        label += ' ' + '{:.2f}'.format(probability * 100) + '%'
    # Create a Rectangle patch
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1,
                             edgecolor=c, facecolor='none')
    add_label_to_plot(ax, c, label, x_min, y_min)
    # Add the patch to the Axes
    ax.add_patch(rect)
    if not axis:
        ax.axis('off')


def plot_image_annotations_simple(image_path: str, title: str = ""):
    """ the image file should finish with: /dataset_name/train_val_test/image"""
    info = image_path.split('/')
    dataset_name, train_val_test, image = info[-3:]
    annotations = helpers.get_annotations_dict(dataset_name, train_val_test)[image]
    labels_map_dict = helpers.get_labels_dict(dataset_name)
    plot_image_annotations(image_path, annotations, labels_map_dict, title=title)


def add_label_to_plot(ax, c, label, x_min, y_min):
    # Add the label text with same color as the corresponding bounding box
    if np.array_equal(c, (0, 0, 0)):
        ax.text(x_min, y_min - 2, s=label, fontsize=5, bbox=dict(facecolor=c, linewidth=0, boxstyle='square,pad=0'),
                color=(1, 1, 1))
    else:
        ax.text(x_min, y_min - 2, s=label, fontsize=5, bbox=dict(facecolor=c, linewidth=0, boxstyle='square,pad=0'))


def plot_images_and_boxes(img, boxes, switch=False, multi=True, i=None, title="", dataset_name=None, colors=None):
    if type(boxes) != np.ndarray:
        boxes = np.array(boxes)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    img_shape = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    if dataset_name:
        labels = []
        labels_map = helpers.get_labels_dict(dataset_name)
        colors = get_colors(boxes, labels)

    if switch:
        boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]

    for bbox in boxes:
        original = bbox
        if multi:
            bbox = [int(elem) for elem in list(bbox[:4] * img_shape)] + ([int(bbox[-1])] if len(bbox) == 5 else [])
        if np.sum(bbox[:4]) == 0:
            continue
        # print("Box added:", bbox, "original", original)
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=1,
                                 edgecolor=colors[labels.index(bbox[-1])],
                                 facecolor='none')
        if dataset_name and len(bbox) == 5:
            add_label_to_plot(ax, colors[labels.index(bbox[-1])], labels_map[str(bbox[-1])], bbox[0], bbox[1])

        plt.scatter(bbox[0], bbox[1], c='r', marker='x')
        plt.scatter(bbox[2], bbox[3], c='g', marker='x')
        ax.add_patch(rect)
    plt.title(title)
    if i is not None:
        plt.savefig('{:03d}.png'.format(i))
    plt.show()
    if i is not None:
        return i + 1
    else:
        return i


def visualize_box(boxes, image, i):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(image)
    for b in boxes:
        ax.add_patch(plt.Polygon(b, fill=None, edgecolor='r', linewidth=3))
        # ax.add_patch(mpatches.Polygon(b, True))
    # ax.autoscale_view()
    plt.savefig('{:03d}.png'.format(i))
    plt.show()
