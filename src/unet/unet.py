import tensorflow as tf


def upsample(filters, size):
    """
    This function defines the upsampling block of the U-Net, consisting of a 
    Transposed Convolution layer followed by a ReLU activation function.

    Parameters:
    - filters (int): The number of filters for the transposed convolution layer.
    - size (int): The kernel size for the transposed convolution layer.

    Returns:
    - result (tf.keras.Sequential): The upsampling block.
    """
    # Initialize the weights of the layer
    initializer = tf.random_normal_initializer(0., 0.02)

    # Define the Sequential model to hold the layers
    result = tf.keras.Sequential()

    # Add a Transposed Convolution layer
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    # Add a ReLU activation function
    result.add(tf.keras.layers.ReLU())

    return result


def Unet(output_channels=2, input_shape=[512, 512, 3]):
    """
    This function defines the U-Net model using MobileNetV2 as the encoder (downsampling stack).

    Parameters:
    - output_channels (int): The number of output channels of the last layer.
    - input_shape (list): The shape of the input images.

    Returns:
    - model (tf.keras.Model): The U-Net model.
    """
    # Load the pre-trained MobileNetV2 model without the top layer
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False)

    # Select the layers that are useful for the skip connections
    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project',
    ]

    # Get the output of the selected layers
    base_model_outputs = [base_model.get_layer(
        name).output for name in layer_names]

    # Define the downsampling stack using the outputs of the selected layers
    down_stack = tf.keras.Model(
        inputs=base_model.input, outputs=base_model_outputs)

    # Set the layers of the downsampling stack as non-trainable
    down_stack.trainable = False

    # Define the upsampling stack
    up_stack = [
        upsample(512, 3),
        upsample(256, 3),
        upsample(128, 3),
        upsample(64, 3),
    ]

    # Define the input layer of the model
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Get the skip connections for the input
    skips = down_stack(inputs)

    # Get the output of the last layer of the downsampling stack
    x = skips[-1]

    # Reverse the order of the layers in the skip connections
    skips = reversed(skips[:-1])

    # Go through the upsampling stack and concatenate with the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # Define the last Transposed Convolution layer
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2, padding='same')

    # Get the output of the last layer
    x = last(x)

    # Apply a Softmax activation function to the output
    x = tf.keras.layers.Softmax(axis=-1)(x)

    # Return the U-Net model
    return tf.keras.Model(inputs=inputs, outputs=x)
