import tensorflow as tf
from tune import generate_dataset, tuning

# Define the paths where the model and feature extractor will be saved
path_to_model = f'../../saved_model/resnet50_.h5'
path_to_feature_extractor = f'../../saved_model/resnet50_feature_extractor_.h5'


def get_resnet():
    """
    Loads the ResNet50 model pre-trained on ImageNet without the top (fully connected) layers.

    Returns:
    - model (tf.keras.Model): The loaded ResNet50 model.
    """
    return tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                   weights='imagenet',
                                                   input_shape=(200, 200, 3),
                                                   pooling="avg",
                                                   classifier_activation="softmax")


# Load the ResNet50 model
model = get_resnet()

# Generate the dataset for tuning
dataset = generate_dataset()

# Perform the tuning on the model using the generated dataset
model, feature_extractor = tuning(model, dataset)

# Save the tuned model and the feature extractor to the specified paths
model.save(path_to_model)
feature_extractor.save(path_to_feature_extractor)
