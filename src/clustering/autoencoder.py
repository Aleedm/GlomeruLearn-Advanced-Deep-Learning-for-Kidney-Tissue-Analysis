import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from clustering import run_clustering

# Define constants
feature_extractor = "autoencoder"
base_url = "autoencoder"
pca_components = 100
path_to_saved_model = "../../saved_model/autoencoder_v2.3_newdata_dense_512.h5"
path_to_dataset = "../../dataset512x512/image_test_autoencoder.npy"
path_to_original_dataset = "../../dataset512x512/image_test_autoencoder.npy"
save_data = False


def get_autoencoder():
    """Load the saved Autoencoder model and return the encoder part."""
    model = tf.keras.models.load_model(path_to_saved_model)
    encoder = model.layers[1]
    return encoder


def load_data():
    """Load and return the dataset and the original dataset."""
    ds = np.load(path_to_dataset)
    return ds, ds


def main():
    """Main function to orchestrate the feature extraction and clustering."""

    # Load dataset and original dataset
    dataset, dataset_original = load_data()

    # Normalize dataset if necessary
    if dataset.max() > 1:
        dataset = dataset.astype('float32') / 255

    # Load the encoder part of the Autoencoder and extract features
    model = get_autoencoder()
    features = model.predict(dataset)

    # Optionally, save the extracted features
    if save_data:
        np.save("../../dataset512x512/glomeruli_features_{}.npy".format(base_url),
                features)
    print(features.shape)
    print(features.max())
    print(features.min())

    # Run clustering on the extracted features
    run_clustering(dataset_original, features,
                   feature_extractor, base_url)

    # Perform PCA on the features and run clustering on the transformed features
    features_pca = PCA(n_components=pca_components).fit_transform(features)
    run_clustering(dataset_original, features_pca,
                   "{}_pca".format(feature_extractor), base_url)


# Execute the main function
main()
