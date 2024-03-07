import numpy as np
from sklearn.model_selection import train_test_split
from data_aug import data_aug_impl

dataset_path = "../../dataset512x512/dataset.npy"
labels_path = "../../dataset512x512/labels.npy"
path_to_save_train_image = "../../dataset512x512/image_train_ss.npy"
path_to_save_train_label = "../../dataset512x512/label_train_ss.npy"
path_to_save_test_image = "../../dataset512x512/image_test_ss.npy"
path_to_save_test_label = "../../dataset512x512/label_test_ss.npy"

dataset = np.load(dataset_path)
labels = np.load(labels_path)

# print the shapes of the dataset and labels
print(f"dataset shape: {dataset.shape}")
print(f"labels shape: {labels.shape}")

image_train, image_test, label_train, label_test = train_test_split(
    dataset, labels, test_size=0.30, random_state=42)

print(f"image_train shape: {image_train.shape}")
print(f"label_train shape: {label_train.shape}")
print(f"image_test shape: {image_test.shape}")
print(f"label_test shape: {label_test.shape}")

image_train_aug, label_train_aug = data_aug_impl(image_train, label_train)

print(f"image_train_aug shape: {image_train_aug.shape}")
print(f"label_train_aug shape: {label_train_aug.shape}")

image_train_aug = np.concatenate((image_train, image_train_aug), axis=0)
label_train_aug = np.concatenate((label_train, label_train_aug), axis=0)

print(f"image_train_aug concatenated shape: {image_train_aug.shape}")
print(f"label_train_aug concatenated shape: {label_train_aug.shape}")

np.save(path_to_save_train_image, image_train_aug)
np.save(path_to_save_train_label, label_train_aug)
np.save(path_to_save_test_image, image_test)
np.save(path_to_save_test_label, label_test)
