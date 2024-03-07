import cv2
from matplotlib import pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import threading


def get_annotations(file_path):
    """Parse XML file to get annotations."""
    tree = ET.parse(file_path)
    root = tree.getroot()

    annotations = []
    for annotation in root.iter('Annotation'):
        name = annotation.get('Name')
        coordinates = []
        for coordinate in annotation.iter('Coordinate'):
            x = float(coordinate.get('X'))
            y = float(coordinate.get('Y'))
            coordinates.append((x, y))
        annotations.append({'name': name, 'coordinates': coordinates})
    return annotations


def is_mostly_white(image, threshold_w=0.85, threshold_p=0.98):
    """Check if the image is mostly white based on thresholds."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixel_threshold = int(threshold_w * 255)
    white_pixels = np.sum(gray_image >= pixel_threshold)
    white_percentage = white_pixels / \
        (gray_image.shape[0] * gray_image.shape[1])
    return white_percentage >= threshold_p, white_percentage


def get_labels(labels, annotations):
    """Fill the labels image based on annotations."""
    for annotation in annotations:
        polygon = np.array([annotation['coordinates']], dtype=np.int32)
        cv2.fillPoly(labels, polygon, 1)


def extrapolate_patches(wsi, annotation, el_width, el_height, output_width, output_height):
    # Get dimensions of the whole slide image
    w, h = wsi.dimensions
    # Initialize a label image with zeros, of the same size as the WSI
    label_image = np.zeros((h, w), dtype=np.uint8)
    # Parse the annotation file to get the annotated regions
    annotations = get_annotatios(annotation)
    # Fill the label image based on the annotations
    get_labels(label_image, annotations)

    # Calculate the number of rows and columns based on the element width and height
    num_rows = h // el_height
    num_cols = w // el_width

    # Initialize lists to store the extracted patches and their corresponding labels
    dataset = []
    labels = []

    # Retrieve the thread name (not used in this snippet)
    thread_name = threading.current_thread().name

    # Read the entire region of the WSI and convert to a NumPy array
    wsi = np.array(wsi.read_region((0, 0), 0, (w, h)))

    # Loop over the rows and columns to extract patches
    for row in range(num_rows):
        for col in range(num_cols):
            # Calculate the coordinates of the top-left and bottom-right corners of the patch
            x = col * el_width
            y = row * el_height
            x_end = x + el_width
            y_end = y + el_height

            # Extract the region and convert it to BGR format
            region = wsi[y: y_end, x: x_end]
            image = cv2.cvtColor(region, cv2.COLOR_RGBA2BGR)

            # Check if the patch is not mostly white
            is_white, p = is_mostly_white(image)
            if not is_white:
                # Resize the patch and its corresponding label to the output dimensions
                r_image = cv2.resize(
                    image, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
                r_label_image = cv2.resize(label_image[y:y_end, x: x_end], (
                    output_width, output_height), interpolation=cv2.INTER_CUBIC)

                # Append the resized patch and label to the lists
                dataset.append(r_image)
                labels.append(r_label_image)

                # If the current patch is not at the border of the WSI, extract additional half-shifted patches
                if not ((col == num_cols-1) or (row == num_rows-1)):
                    # Calculate coordinates for half-shifted patches
                    x_h = x + el_width // 2
                    x_v = x
                    x_d = x + el_width // 2
                    y_h = y
                    y_v = y + el_height // 2
                    y_d = y + el_height // 2

                    # Extract half-shifted patches
                    region_h = wsi[y_h: y_h + el_height, x_h: x_h + el_width]
                    region_v = wsi[y_v: y_v + el_height, x_v: x_v + el_width]
                    region_d = wsi[y_d: y_d + el_height, x_d: x_d + el_width]

                    # Convert to BGR format
                    image_h = cv2.cvtColor(
                        np.array(region_h), cv2.COLOR_RGBA2BGR)
                    image_v = cv2.cvtColor(
                        np.array(region_v), cv2.COLOR_RGBA2BGR)
                    image_d = cv2.cvtColor(
                        np.array(region_d), cv2.COLOR_RGBA2BGR)

                    # Check if the half-shifted patches are not mostly white
                    is_white_h, _ = is_mostly_white(image_h)
                    is_white_v, _ = is_mostly_white(image_v)
                    is_white_d, _ = is_mostly_white(image_d)

                    # Resize and append non-white half-shifted patches and their labels to the lists
                    if not is_white_h:
                        r_image = cv2.resize(
                            image_h, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
                        r_label_image = cv2.resize(
                            label_image[y_h: y_h+el_height, x_h: x_h+el_width], (output_width, output_height), interpolation=cv2.INTER_CUBIC)
                        dataset.append(r_image)
                        labels.append(r_label_image)

                    if not is_white_v:
                        r_image = cv2.resize(
                            image_v, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
                        r_label_image = cv2.resize(
                            label_image[y_v: y_v+el_height, x_v: x_v+el_width], (output_width, output_height), interpolation=cv2.INTER_CUBIC)
                        dataset.append(r_image)
                        labels.append(r_label_image)

                    if not is_white_d:
                        r_image = cv2.resize(
                            image_d, (output_width, output_height), interpolation=cv2.INTER_CUBIC)
                        r_label_image = cv2.resize(
                            label_image[y_d: y_d+el_height, x_d: x_d+el_width], (output_width, output_height), interpolation=cv2.INTER_CUBIC)
                        dataset.append(r_image)
                        labels.append(r_label_image)

    # Return the list of patches and their corresponding labels
    return dataset, labels
