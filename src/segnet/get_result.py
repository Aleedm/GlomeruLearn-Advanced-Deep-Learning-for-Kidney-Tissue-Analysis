import tensorflow as tf
import numpy as np

# Define the paths to the model and datasets
model_path = '../../saved_model/segnet_V1.2__balanced_dataset_'
image_validation_path = '../../dataset512x512/image_validation_balanced.npy'
image_test_path = '../../dataset512x512/image_test_balanced.npy'

# Define the paths where the resulting dataset and labels will be saved
path_to_save_dataset = '../../dataset512x512/image_test_step_2.npy'
path_to_save_labels = '../../dataset512x512/label_test_step_2.npy'

# Load the trained SegNet model
model = tf.keras.models.load_model(model_path)

# Load the test and validation datasets and normalize the pixel values
data_test = np.load(image_test_path) / 255
data_validation = np.load(image_validation_path) / 255

# Print the shapes of the loaded datasets for verification
print(f"Dataset test shape {data_test.shape}")
print(f"Dataset validation shape {data_validation.shape}")

# Concatenate the test and validation datasets along the first axis (batch size)
ds = np.append(data_test, data_validation, axis=0)

# Print the shape of the concatenated dataset
print(f"Dataset shape {ds.shape}")

# Use the model to predict the labels for the concatenated dataset
res_lab = model.predict(ds)

# Print the shape of the resulting labels
print(f"Label shape {res_lab.shape}")

# Save the concatenated dataset and the resulting labels to the specified paths
np.save(path_to_save_dataset, ds)
np.save(path_to_save_labels, res_lab)
