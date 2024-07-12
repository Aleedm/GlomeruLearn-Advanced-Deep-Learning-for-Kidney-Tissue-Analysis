import cv2
import numpy as np
import tensorflow as tf
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def glomeruli_crop(glomeruli, glomeruli_labels):
    """
    Function to crop images containing glomeruli based on the connected components in the labels.
    It resizes the cropped images and returns arrays of cropped images, corresponding labels, original images, and original labels.
    """
    result = []  # To store cropped and resized images
    original_images = []  # To store original images
    original_labels = []  # To store original labels 
    result_labels = []  # To store cropped and resized labels

    # Iterate through all the images and corresponding labels
    for j in range(0, len(glomeruli)):
        num_labels, labels = cv2.connectedComponents(
            glomeruli_labels[j], connectivity=8)

        # Iterate through all connected components (excluding the background)
        for i in range(1, num_labels):
            image = glomeruli[j]

            # Find the bounding box around the connected component
            yrow, xcol = np.where(labels == i)
            yrowmin, yrowmax = np.min(yrow), np.max(yrow)
            xcolmin, xcolmax = np.min(xcol), np.max(xcol)

            # Determine the side length of the square bounding box
            side = max(yrowmax - yrowmin + 1, xcolmax - xcolmin + 1)

            # Check if the bounding box is large enough
            if (side >= 20):
                # Adjust the bounding box to be square and within the image boundaries
                yrowmin = max(
                    0, yrowmin - (side - (yrowmax - yrowmin + 1)) // 2)
                yrowmax = min(image.shape[0], yrowmin + side)
                xcolmin = max(
                    0, xcolmin - (side - (xcolmax - xcolmin + 1)) // 2)
                xcolmax = min(image.shape[1], xcolmin + side)

                # Extract the square cropped image and corresponding label
                cropped_image = image[yrowmin:yrowmax, xcolmin:xcolmax]
                cropped_label = labels[yrowmin:yrowmax, xcolmin:xcolmax] == i

                # Keep the cropped image if it contains a sufficient proportion of glomeruli
                if np.mean(cropped_label) >= 0.6:
                    resized_image = cv2.resize(
                        cropped_image, (200, 200), interpolation=cv2.INTER_AREA)
                    resized_label = cv2.resize(cropped_label.astype(
                        np.uint8), (200, 200), interpolation=cv2.INTER_NEAREST)
                    result.append(resized_image)
                    result_labels.append(resized_label)
                    original_images.append(image)
                    original_labels.append(labels)

    return np.array(result), np.array(result_labels), np.array(original_images), np.array(original_labels)


def glomeruli_crop_dark_backgroung_super_cool(glomeruli, glomeruli_labels):
    """
    Function to mask glomeruli images with their corresponding labels.
    The function returns arrays of masked images and labels.
    """
    result = []  # To store masked images
    result_labels = []  # To store labels

    # Set all non-zero label values to 1
    glomeruli_labels[glomeruli_labels != 0] = 1

    # Iterate through all the images and corresponding labels to apply the mask
    for i in range(0, len(glomeruli)):
        result_image = cv2.merge([channel * glomeruli_labels[i]
                                 for channel in cv2.split(glomeruli[i])])
        result.append(result_image)
        result_labels.append(glomeruli_labels[i])

    return np.array(result), np.array(result_labels)


def process_data(image):
    """
    Function to normalize the pixel values of the input image.
    Returns the normalized image.
    """
    return tf.cast(image, tf.float32) / 255, tf.cast(image, tf.float32) / 255


def data_augment():
    """
    Function to define a sequence of augmentation techniques to be applied to the images.
    Returns the augmentation sequence.
    """
    return iaa.Sequential([
        iaa.Dropout((0, 0.05)),  # Randomly remove pixels
        iaa.Affine(rotate=(-30, 30)),  # Rotate image within a range
        iaa.Fliplr(0.5),  # Horizontal flip with 50% probability
        # Random crop while keeping the original size
        iaa.Crop(percent=(0, 0.2), keep_size=True),
        iaa.WithBrightnessChannels(iaa.Add((-50, 50))),  # Adjust brightness
        # Convert to grayscale with varying strength
        iaa.Grayscale(alpha=(0.0, 0.5)),
        # Adjust gamma contrast
        iaa.GammaContrast((0.5, 2.0), per_channel=True),
        iaa.PiecewiseAffine(scale=(0.01, 0.1)),  # Apply local distortions
    ], random_order=True)


def data_aug_impl_no_label(image_train, n=1):
    """
    Function to augment a given dataset of images without labels.
    The augmented images are appended to the original dataset.
    Returns the augmented dataset.
    """
    da = data_augment()
    for i in range(n):
        augmented_images = da(images=image_train.copy())
        image_train = np.append(image_train, augmented_images, axis=0)
    return image_train


def data_aug_impl(shape_dataset, image_train, label_train):
    """
    Function to augment a given dataset of images with their corresponding labels.
    The augmented images and labels are appended to the original datasets.
    Returns the augmented datasets of images and labels.
    """
    da = data_augment()
    segmented_label_train = [SegmentationMapsOnImage(
        label, shape=shape_dataset) for label in label_train]
    augmented_images, augmented_labels = da(
        images=image_train.copy(), segmentation_maps=segmented_label_train)
    image_train = np.append(image_train, augmented_images, axis=0)
    label_train = np.append(label_train, np.array(
        [label.get_arr() for label in augmented_labels]), axis=0)
    return image_train, label_train
