import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np


def data_augment():
    return iaa.Sequential([
        # iaa.Dropout((0, 0.05)),  # Remove random pixel
        # iaa.Affine(rotate=(-30, 30)),  # Rotate between -30 and 30 degreed
        iaa.Fliplr(0.5),  # Flip with 0.5 probability
        iaa.Crop(percent=(0, 0.2), keep_size=True),  # Random crop
        # Add -50 to 50 to the brightness-related channels of each image
        iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
        # Change images to grayscale and overlay them with the original image by varying strengths, effectively removing 0 to 50% of the color
        iaa.Grayscale(alpha=(0.0, 0.5)),
        # Add random value to each pixel
        iaa.GammaContrast((0.5, 2.0), per_channel=True),
        # Local distortions of images by moving points around
        iaa.PiecewiseAffine(scale=(0.01, 0.1)),
    ], random_order=True)


def data_aug_impl(image_train, label_train):
    da = data_augment()
    segmented_label_train = [SegmentationMapsOnImage(
        label, shape=image_train[0].shape) for label in label_train]
    image_train_copy = image_train.copy()
    augmented_images, augmented_labels = da(
        images=image_train_copy, segmentation_maps=segmented_label_train)
    augmented_labels = np.array(
        [label.get_arr() for label in augmented_labels])
    return augmented_images, augmented_labels


def data_aug_thread(image, label):
    image_train, label_train = data_aug_impl(image, label)
    return image_train, label_train
