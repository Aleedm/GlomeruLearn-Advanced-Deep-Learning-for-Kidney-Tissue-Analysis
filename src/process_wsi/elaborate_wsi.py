import concurrent.futures
import glob
import numpy as np
from thread import process_svs_file  # Import the process_svs_file function

import openslide

# Define paths and parameters for image processing
path_to_images = "../slides/"
path_to_annotations = "../annotations/"
el_width = 2000
el_height = 2000
output_width = 512
output_height = 512

# Initialize lists to hold the dataset and labels
dataset = []
labels = []

# Retrieve the list of all .svs files in the specified directory
svs_files = glob.glob(path_to_images + "*.svs")

# Define the number of threads for concurrent execution
num_threads = 9
# Create a ThreadPoolExecutor with the specified number of threads
executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

# Initialize a list to hold future objects returned by the executor
futures = []

# Loop over each .svs file and submit it for processing using the process_svs_file function
for svs_file in svs_files:
    future = executor.submit(process_svs_file, svs_file, svs_file, path_to_annotations,
                             path_to_images, el_width, el_height, output_width, output_height)
    futures.append(future)  # Append the future object to the list of futures

# Wait for all futures to complete
concurrent.futures.wait(futures)

# Initialize lists to hold the resulting dataset and labels
dataset = []
labels = []
# Retrieve the results from the completed futures and extend the dataset and labels lists
for future in futures:
    d, l = future.result()
    dataset.extend(d)
    labels.extend(l)

# Convert the lists of dataset and labels to numpy arrays
dataset = np.array(dataset)
labels = np.array(labels)

# Save the resulting dataset and labels as .npy files
np.save('../slides/dataset.npy', dataset)
np.save('../annotations/labels.npy', labels)
