import numpy as np
import tensorflow as tf
from data_processing import get_image_with_glomeruli, get_image_with_no_glomeruli, resize_images, shuffle

# Define training parameters
epoch = 10
batch_size = 8
path_dataset = '../../dataset512x512/image_train_augmented_balanced.npy'
path_label = '../../dataset512x512/label_train_augmented_balanced.npy'


def load_data():
    """
    Loads the dataset and corresponding labels from the predefined paths.

    Returns:
    - ds (np.array): The loaded dataset of images.
    - lb (np.array): The loaded array of labels.
    """
    ds = np.load(path_dataset)
    lb = np.load(path_label)
    return ds, lb


def preprocess_data(ds, lb):
    """
    Preprocesses the dataset by resizing the images and creating labels for images with and without glomeruli.

    Parameters:
    - ds (np.array): The dataset of images.
    - lb (np.array): The corresponding labels for the images.

    Returns:
    - dataset (np.array): The preprocessed dataset of images.
    - labels (np.array): The corresponding labels for the preprocessed images.
    """
    # Extract and resize images with glomeruli, and assign them label 1
    glomeruli = get_image_with_glomeruli(ds, lb)
    glomeruli = resize_images(glomeruli)
    glomeruli_labels = np.ones(glomeruli.shape[0])

    # Extract and resize images without glomeruli, and assign them label 0
    no_glomeruli = get_image_with_no_glomeruli(ds, lb)
    no_glomeruli = resize_images(no_glomeruli)
    no_glomeruli_labels = np.zeros(no_glomeruli.shape[0])

    # Combine the processed images and labels, and shuffle them
    dataset = np.append(glomeruli, no_glomeruli, axis=0)
    labels = np.append(glomeruli_labels, no_glomeruli_labels, axis=0)
    dataset, labels = shuffle(dataset, labels)

    return dataset, labels


def data_generator(ds, lb):
    """
    Generator function to yield images and labels one at a time.

    Parameters:
    - ds (np.array): The dataset of images.
    - lb (np.array): The corresponding labels for the images.
    """
    for d, l in zip(ds, lb):
        yield (d, l)


def process_data(image, label):
    """
    Preprocesses the image and label for training.

    Parameters:
    - image (np.array): The input image.
    - label (int): The corresponding label for the image.

    Returns:
    - image (tf.Tensor): The preprocessed image.
    - label (int): The label.
    """
    return tf.cast(image, tf.float32)/255, label


def load_generator(ds, lb):
    """
    Creates a tf.data.Dataset from the provided images and labels.

    Parameters:
    - ds (np.array): The dataset of images.
    - lb (np.array): The corresponding labels for the images.

    Returns:
    - dataset (tf.data.Dataset): The created dataset.
    """
    output_shapes_train = (tf.TensorSpec(shape=(200, 200, 3), dtype=tf.float32),
                           tf.TensorSpec(shape=(), dtype=tf.int64))

    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(ds, lb), output_signature=output_shapes_train)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.map(process_data, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset


def generate_dataset():
    """
    Loads, preprocesses, and generates a tf.data.Dataset from the image data and labels.

    Returns:
    - dataset (tf.data.Dataset): The generated dataset.
    """
    ds, lb = load_data()
    ds, lb = preprocess_data(ds, lb)
    dataset = load_generator(ds, lb)
    return dataset


def tuning(model, dataset):
    """
    Adds fully connected layers to the provided model and trains it on the given dataset.

    Parameters:
    - model (tf.keras.Model): The base model.
    - dataset (tf.data.Dataset): The dataset for training the model.

    Returns:
    - model (tf.keras.Model): The trained model.
    - feature_extractor (tf.keras.Model): The feature extractor model obtained from the trained model.
    """
    # Add fully connected layers to the model
    x = model(model.input)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    x = tf.keras.layers.Dense(4096, activation="relu")(x)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)
    model = tf.keras.Model(model.input, outputs)

    # Set the first 10 layers to non-trainable
    for i, layer in enumerate(model.layers):
        if i > 10:
            layer.trainable = True
        else:
            layer.trainable = False

    # Compile and train the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(dataset, epochs=epoch, batch_size=batch_size)

    # Create a feature extractor model from the trained model
    layer_to_remove = model.layers[-4].name
    feature_extractor = tf.keras.models.Model(
        inputs=model.input, outputs=model.get_layer(layer_to_remove).output)

    return model, feature_extractor
