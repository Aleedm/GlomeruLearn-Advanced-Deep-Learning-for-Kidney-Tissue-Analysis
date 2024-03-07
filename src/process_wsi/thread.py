import threading
import openslide
from utils import extrapolate_patches
import numpy as np


def process_svs_file(svs_file, path_to_annotations, path_to_images, el_width, el_height, output_width, output_height):
    # Retrieve the name of the current thread
    thread_name = threading.current_thread().name

    # Construct the path to the annotation file corresponding to the input svs_file
    annotation = path_to_annotations + \
        svs_file[len(path_to_images):-4] + ".xml"

    # Open the svs file using openslide
    wsi = openslide.OpenSlide(svs_file)

    # Call the extrapolate_patches function to extract patches and corresponding labels from the whole slide image
    d, l = extrapolate_patches(
        wsi, annotation, el_width, el_height, output_width, output_height)

    # Save the extracted patches and labels to .npy files
    np.save('../slides/' +
            svs_file[len(path_to_images):-4] + '.npy', np.array(d))
    np.save('../annotations/' +
            svs_file[len(path_to_images):-4] + '_label.npy', np.array(l))

    # Return the patches and labels
    return d, l
