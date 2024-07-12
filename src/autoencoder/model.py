import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPool2D
import numpy as np


def build_autoencoder(img_shape, code_size=1024):
    """
    Function to build an autoencoder model consisting of an encoder and a decoder.

    :param img_shape: Shape of the input images.
    :param code_size: Dimensionality of the encoded representation (latent space).
    :return: A tuple containing the encoder and decoder models.
    """

    # Calculate the shape of the representation before flattening in the encoder.
    pre_flatten_shape = (img_shape[0] // 8, img_shape[1] // 8, code_size) 

    # Define the encoder model.
    encoder = tf.keras.Sequential([
        # Convolutional layer with ReLU activation and max pooling.
        Conv2D(code_size//4, 3, padding="same",
               input_shape=img_shape, activation='relu'),
        MaxPool2D(2),

        # Another convolutional layer with ReLU activation and max pooling.
        Conv2D(code_size//2, 3, padding="same", activation='relu'),
        MaxPool2D(2),

        # Final convolutional layer with ReLU activation and max pooling.
        Conv2D(code_size, 3, padding="same", activation='relu'),
        MaxPool2D(2),

        # Flatten the representation and use a dense layer to obtain the encoded representation.
        Flatten(),
        Dense(code_size, activation='relu')
    ])

    # Define the decoder model.
    decoder = tf.keras.Sequential([
        # Fully connected layer to inflate the representation back to pre_flatten_shape.
        Dense(np.prod(pre_flatten_shape),
              activation='relu', input_shape=(code_size,)),
        Reshape(pre_flatten_shape),

        # Transposed convolutional layers to upsample the representation back to the original image size.
        Conv2DTranspose(code_size, 5, padding="same",
                        strides=2, activation='relu'),
        Conv2DTranspose(code_size//2, 3, padding="same",
                        strides=2, activation='relu'),
        Conv2DTranspose(code_size//4, 2, padding="same",
                        strides=2, activation='relu'),

        # Final transposed convolutional layer to obtain the reconstructed image.
        Conv2DTranspose(3, 3, activation='sigmoid', padding="same")
    ])

    return encoder, decoder
