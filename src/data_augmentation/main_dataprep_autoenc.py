import numpy as np
from data_proc import glomeruli_crop
from sklearn.model_selection import train_test_split
from data_bal import augment_images, shuffle

dataset_path = "../../dataset512x512/dataset.npy"
labels_path = "../../dataset512x512/labels.npy"
path_to_save_train_image = "../../dataset512x512/image_train_autoencoder.npy"
path_to_save_train_label = "../../dataset512x512/label_train_autoencoder.npy"
path_to_save_test_image = "../../dataset512x512/image_test_autoencoder.npy"
path_to_save_test_label = "../../dataset512x512/label_test_autoencoder.npy"

dataset = np.load(dataset_path)
labels = np.load(labels_path)

# print the shapes of the dataset and labels
print(f"dataset shape: {dataset.shape}")
print(f"labels shape: {labels.shape}")

glom, glom_l, _, _ = glomeruli_crop(dataset, labels)

# print the shapes of the glomeruli and glomeruli labels
print(f"glomeruli shape: {glom.shape}")
print(f"glomeruli labels shape: {glom_l.shape}")

glom_aug, glom_l_aug = augment_images(glom, glom_l, augmentation_factor=1)

print(f"glomeruli augmented shape: {glom_aug.shape}")
print(f"glomeruli labels augmented shape: {glom_l_aug.shape}")

glom_balanced = np.concatenate((glom, glom_aug), axis=0)
glom_l_balanced = np.concatenate((glom_l, glom_l_aug), axis=0)
glom_balanced, glom_l_balanced = shuffle(glom_balanced, glom_l_balanced)

print(f"glomeruli balanced shape: {glom_balanced.shape}")
print(f"glomeruli labels balanced shape: {glom_l_balanced.shape}")

image_train, image_test, label_train, label_test = train_test_split(
    glom_balanced, glom_l_balanced, test_size=0.30, random_state=42)


print(f"image_train shape: {image_train.shape}")
print(f"label_train shape: {label_train.shape}")
print(f"image_test shape: {image_test.shape}")
print(f"label_test shape: {label_test.shape}")

np.save(path_to_save_train_image, image_train)
np.save(path_to_save_train_label, label_train)
np.save(path_to_save_test_image, image_test)
np.save(path_to_save_test_label, label_test)
