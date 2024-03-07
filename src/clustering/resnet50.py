import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from clustering import run_clustering

# Define constants
feature_extractor = "resnet50"
base_url = "resnet50_tuned"
pca_components = 100
batch_size = 8
save_data = False
path_to_saved_resnet50 = "../../saved_model/resnet50_feature_extractor_.h5"
path_to_dataset = "../../dataset512x512/image_test_step_2d_cropped_darked.npy"
path_to_original_dataset = "../../dataset512x512/image_test_step_2d_cropped.npy"


def get_resnet50():
    """Load and return the saved ResNet50 model."""
    return tf.keras.models.load_model(path_to_saved_resnet50)


def load_data():
    """Load and return the dataset and the original dataset."""
    ds = np.load(path_to_dataset)
    ds_o = np.load(path_to_original_dataset)
    return ds, ds_o


def main():
    """Main function to orchestrate the feature extraction and clustering."""

    # Load dataset and original dataset
    dataset, dataset_original = load_data()
    print(dataset.shape)

    # Load ResNet50 model and extract features
    model = get_resnet50()
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
