import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.manifold import Isomap, TSNE
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import random


def plot_scatter(x, y, labels, title):
    """
    Plots a scatter plot with the given x and y coordinates, colored by labels.
    """
    cmap = plt.cm.get_cmap('tab20', len(set(labels)))
    plt.scatter(x, y, c=labels, cmap=cmap)
    plt.title(title)
    plt.show()


def plot_results(transformer, kmeans, best_params, transformer_type, dataset, title, url):
    """
    Visualizes the results of clustering using different techniques for dimensionality reduction.
    Saves the resulting plots to the specified URL with the given title as a filename prefix.
    """
    labels = kmeans.labels_
    cmap = plt.cm.get_cmap('viridis', len(np.unique(labels)))

    if transformer_type in ['kmeans', 'spectral']:
        # Apply Linear Discriminant Analysis, Partial Least Squares Regression,
        # and Neighborhood Components Analysis for dimensionality reduction and plotting.
        for method, cls in zip(['lda', 'pls', 'nca'], [LinearDiscriminantAnalysis, PLSRegression, NeighborhoodComponentsAnalysis]):
            reducer = cls(n_components=2)
            dataset_reduced = reducer.fit_transform(
                dataset, labels) if method != 'pls' else reducer.fit_transform(dataset, labels)[0]
            plt.figure()
            plt.scatter(dataset_reduced[:, 0],
                        dataset_reduced[:, 1], c=labels, cmap=cmap)
            plt.savefig(f"{url}{title}_{transformer_type}_{method}.png")
            plt.clf()
    else:
        n_components = best_params[f'{transformer_type}__n_components'] if transformer_type == 'isomap' else best_params['n_components']
        if n_components <= 3:
            # If the number of components is 2 or 3, visualize the data in 2D or 3D space.
            projection = transformer.fit_transform(dataset)
            fig = plt.figure()
            if n_components == 2:
                plt.scatter(projection[:, 0],
                            projection[:, 1], c=labels, cmap=cmap)
            else:
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(projection[:, 0], projection[:, 1],
                           projection[:, 2], c=labels, cmap=cmap)
            plt.savefig(f"{url}{title}_{transformer_type}_{n_components}D.png")
            plt.clf()
        else:
            # If the number of components is more than 3, reduce it to 2 and visualize in 2D space.
            transformer.set_params(n_components=2)
            projection = transformer.fit_transform(dataset)
            plt.scatter(projection[:, 0],
                        projection[:, 1], c=labels, cmap=cmap)
            plt.savefig(
                f"{url}{title}_{transformer_type}_2D_n-comp-more-then-2.png")
            plt.clf()


def plot_images(dataset, labels, title, url):
    """
    Displays and saves a grid of images from the dataset, grouped by the clustering labels.
    """
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    fig, axes = plt.subplots(10, num_labels, figsize=(num_labels * 3, 30))

    for i, label in enumerate(unique_labels):
        images = dataset[labels == label]
        images_to_plot = random.sample(list(images), min(10, len(images)))
        for j, img in enumerate(images_to_plot):
            axes[j, i].imshow(img, cmap='gray')
            axes[j, i].axis('off')
        axes[0, i].set_title(f'Label {label}')
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{url}{title}_images.png")
