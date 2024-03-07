import numpy as np
from sklearn.model_selection import train_test_split
import concurrent.futures
from data_bal import filter_images, augment_images, shuffle
from data_aug import data_aug_thread

path_to_dataset = "../../dataset512x512/dataset.npy"
path_to_labels = "../../dataset512x512/labels.npy"

path_to_save_o_image_train = "../../dataset512x512/image_train_balanced.npy"
path_to_save_o_label_train = "../../dataset512x512/label_train_balanced.npy"

path_to_save_image_train = "../../dataset512x512/image_train_augmented_balanced.npy"
path_to_save_label_train = "../../dataset512x512/label_train_augmented_balanced.npy"

path_to_save_image_validation = "../../dataset512x512/image_validation_balanced.npy"
path_to_save_label_validation = "../../dataset512x512/label_validation_balanced.npy"

path_to_save_image_test = "../../dataset512x512/image_test_balanced.npy"
path_to_save_label_test = "../../dataset512x512/label_test_balanced.npy"


print("Loading dataset and labels 512x512")

dataset = np.load(path_to_dataset)
labels = np.load(path_to_labels)

print(
    f"Dataset and labels loaded\nDataset shape {dataset.shape} \nLabels shape {labels.shape}")

dataset_filtered, labels_filtered = filter_images(dataset, labels)
print(
    f"Dataset and labels filtered\ndataset_filtered shape {dataset_filtered.shape} \nlabels_filtered shape {labels_filtered.shape}")

dataset_aug, labels_aug = augment_images(dataset_filtered, labels_filtered, augmentation_factor=1)
print(
    f"Dataset and labels augmented\ndataset_aug shape {dataset_aug.shape} \nlabels_aug shape {labels_aug.shape}")

dataset_balanced = np.concatenate((dataset_filtered, dataset_aug), axis=0)
labels_balanced = np.concatenate((labels_filtered, labels_aug), axis=0)

dataset_balanced, labels_balanced = shuffle(dataset_balanced, labels_balanced)

print(
    f"Dataset and labels balanced\ndataset_balanced shape {dataset_balanced.shape} \nlabels_balanced shape {labels_balanced.shape}")

image_train, image_vt, label_train, label_vt = train_test_split(
    dataset_balanced, labels_balanced, test_size=0.30, random_state=42)
image_validation, image_test, label_validation, label_test = train_test_split(
    image_vt, label_vt, test_size=0.33, random_state=42)

print("Dataset and labels splitted in train, validation and test set\n" +
      f"image_train shape {image_train.shape} - label_train shape {label_train.shape}" +
      f"image_validation shape {image_validation.shape} - label_validation shape {label_validation.shape}" +
      f"image_test shape {image_test.shape} - image_test shape {label_test.shape}")

np.save(path_to_save_o_image_train, image_train)
np.save(path_to_save_o_label_train, label_train)

np.save(path_to_save_image_validation, image_validation)
np.save(path_to_save_label_validation, label_validation)
#
np.save(path_to_save_image_test, image_test)
np.save(path_to_save_label_test, label_test)
#
num_threads = 3  # Numero di thread da utilizzare
executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)
futures = []

for i in range(0, 3):
   future = executor.submit(data_aug_thread, image_train, label_train)
   futures.append(future)

concurrent.futures.wait(futures)

for future in futures:
   im, l = future.result()
   image_train = np.append(image_train, im, axis=0)
   label_train = np.append(label_train, l, axis=0)

print("Applied data agumentation to train set\n" +
     f"image_train augmented shape {image_train.shape} - label_train augmented shape {label_train.shape}")

np.save(path_to_save_image_train, image_train)
np.save(path_to_save_label_train, label_train)
