import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def filter_images(images, labels):
    # Identify indices of images with only class 0 pixels
    only_class_0_indices = np.where(np.sum(labels, axis=(1, 2)) == 0)[0]
    # Randomly select half of them for removal
    indices_to_remove = np.random.choice(
        only_class_0_indices, size=len(only_class_0_indices) // 2, replace=False)

    # Create a mask to filter out the selected images
    mask = np.ones(len(images), dtype=bool)
    mask[indices_to_remove] = False

    # Return the filtered images and labels
    return images[mask], labels[mask]


def augment_images(images, labels, augmentation_factor=1):
    # Define the augmentation pipeline
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        # iaa.Affine(rotate=(-45, 45)),
        iaa.LinearContrast((0.95, 1.05)),
    ])

    augmented_images = []
    augmented_labels = []

    # Iterate over images and labels and apply augmentation
    for image, label in zip(images, labels):
        if np.sum(label) > 0:  # Check if the image contains class 1 pixels
            segmap = SegmentationMapsOnImage(label, shape=image.shape)
            for _ in range(augmentation_factor):
                image_aug, segmap_aug = seq(
                    image=image, segmentation_maps=segmap)
                augmented_images.append(image_aug)
                augmented_labels.append(segmap_aug.get_arr())
        # else:
        #    augmented_images.append(image)
        #    augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)


def shuffle(images, labels):
    shuffled_indices = np.arange(len(images))
    np.random.shuffle(shuffled_indices)
    images = images[shuffled_indices]
    labels = labels[shuffled_indices]

    return images, labels
