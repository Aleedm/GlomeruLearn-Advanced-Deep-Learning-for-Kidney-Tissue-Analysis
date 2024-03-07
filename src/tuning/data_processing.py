import numpy as np
import cv2


def get_image_with_glomeruli(dataset, labels):
    """
    Extracts images from the dataset that contain glomeruli.

    Parameters:
    - dataset (np.array): The array of images in the dataset.
    - labels (np.array): The corresponding labels for the images in the dataset.

    Returns:
    - (np.array): An array of images that contain glomeruli.
    """
    return np.array([dataset[i] for i in range(len(dataset)) if (1 in labels[i])])


def get_image_with_glomeruli_labels(dataset, labels):
    """
    Extracts labels corresponding to images from the dataset that contain glomeruli.

    Parameters:
    - dataset (np.array): The array of images in the dataset.
    - labels (np.array): The corresponding labels for the images in the dataset.

    Returns:
    - (np.array): An array of labels corresponding to images that contain glomeruli.
    """
    return np.array([labels[i] for i in range(len(dataset)) if (1 in labels[i])])


def get_image_with_no_glomeruli(dataset, labels):
    """
    Extracts images from the dataset that do not contain glomeruli.

    Parameters:
    - dataset (np.array): The array of images in the dataset.
    - labels (np.array): The corresponding labels for the images in the dataset.

    Returns:
    - (np.array): An array of images that do not contain glomeruli.
    """
    return np.array([dataset[i] for i in range(len(dataset)) if not (1 in labels[i])])


def get_image_with_no_glomeruli_labels(dataset, labels):
    """
    Extracts labels corresponding to images from the dataset that do not contain glomeruli.

    Parameters:
    - dataset (np.array): The array of images in the dataset.
    - labels (np.array): The corresponding labels for the images in the dataset.

    Returns:
    - (np.array): An array of labels corresponding to images that do not contain glomeruli.
    """
    return np.array([labels[i] for i in range(len(dataset)) if not (1 in labels[i])])


def shuffle(images, labels):
    """
    Shuffles the order of images and corresponding labels in the dataset.

    Parameters:
    - images (np.array): The array of images in the dataset.
    - labels (np.array): The corresponding labels for the images in the dataset.

    Returns:
    - images (np.array): The shuffled array of images.
    - labels (np.array): The corresponding shuffled labels for the images.
    """
    shuffled_indices = np.arange(len(images))
    np.random.shuffle(shuffled_indices)
    images = images[shuffled_indices]
    labels = labels[shuffled_indices]

    return images, labels


def resize_images(images, size=(200, 200)):
    """
    Resizes the images in the dataset to the specified size.

    Parameters:
    - images (np.array): The array of images to be resized.
    - size (tuple): The desired size for the resized images. Default is (200, 200).

    Returns:
    - (np.array): An array of resized images.
    """
    return np.array([cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC) for img in images])
