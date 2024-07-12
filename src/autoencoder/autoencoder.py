import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# Importing the build_autoencoder function from the model module
from model import build_autoencoder

# Setting hyperparameters and paths
latent_space_dim = 512
version = "2.3_newdata"
batch_size = 8
epochs = 500
IMG_SHAPE = (200, 200, 3)
dataset_path = '../../dataset512x512/image_train_autoencoder.npy'
test_set_path = '../../dataset512x512/image_test_autoencoder.npy'
path_to_save_model = f'../../saved_model/autoencoder_v{version}_dense_{latent_space_dim}.h5'
path_to_save_decoded_image = f"../../images_autoencoder/decoded_imgs_v{version}_dense_{latent_space_dim}.npy"


def data_generator():
    """
    Generator function to yield training samples.
    """
    data = np.load(dataset_path)
    index_90_percent = int(len(data) * 0.90)
    data = data[:index_90_percent]
    for d in data:
        yield d, d  # Input and target are the same for autoencoder


def val_data_generator():
    """
    Generator function to yield validation samples.
    """
    val_data = np.load(dataset_path)
    index_90_percent = int(len(val_data) * 0.90)
    val_data = val_data[index_90_percent:]
    for d in val_data:
        yield d, d


def process_data(image, i):
    """
    Function to preprocess the data.
    """
    return tf.cast(image, tf.float32)/255, tf.cast(i, tf.float32)/255


def load_generator():
    """
    Function to load training and validation datasets using generator functions.
    """
    output_shapes_train = (tf.TensorSpec(shape=(200, 200, 3), dtype=tf.int32),
                           tf.TensorSpec(shape=(200, 200, 3), dtype=tf.int32))
    dataset = tf.data.Dataset.from_generator(
        data_generator, output_signature=output_shapes_train).batch(batch_size, drop_remainder=True).map(
        process_data, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_generator(
        val_data_generator, output_signature=output_shapes_train).batch(batch_size, drop_remainder=True).map(
        process_data, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset, val_dataset


def build_and_compile_model(img_shape, code_size):
    """
    Function to build and compile the autoencoder model. 
    """
    encoder, decoder = build_autoencoder(img_shape, code_size)
    inp = Input(img_shape)
    code = encoder(inp)
    reconstruction = decoder(code)
    metrics = ["accuracy", "Precision", "Recall", "FalseNegatives",
               "FalsePositives", "TrueNegatives", "TruePositives"]
    autoencoder = tf.keras.Model(inputs=inp, outputs=reconstruction)
    autoencoder.compile(optimizer='adamax', loss='mse', metrics=metrics)
    return autoencoder


def train_model(model, ds, val_ds):
    """
    Function to train the autoencoder model.
    """
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        path_to_save_model, save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(
        ds, epochs=epochs, batch_size=batch_size, validation_data=val_ds,
        callbacks=[early_stopping, model_checkpoint])
    return model, history


def save_decoded_images(model):
    """
    Function to save the decoded (reconstructed) images.
    """
    ds_test = np.load(test_set_path)
    predicted = model.predict(ds_test)
    decoded_imgs = (predicted * 255).astype(int)
    np.save(path_to_save_decoded_image, decoded_imgs)


def main():
    """
    Main function to execute the training process.
    """
    ds, val_ds = load_generator()
    autoencoder = build_and_compile_model(IMG_SHAPE, latent_space_dim)
    autoencoder, history = train_model(autoencoder, ds, val_ds)
    save_decoded_images(autoencoder)


if __name__ == "__main__":
    main()
