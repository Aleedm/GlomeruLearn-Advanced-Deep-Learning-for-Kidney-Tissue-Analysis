from isomap import define_best_params_isomap, best_isomap_kmeans
from tsne import define_best_params_tsne, best_tsne_kmeans
from kmeans import define_best_params_kmeans
from sklearn.cluster import KMeans
from plot import plot_results, plot_images
from joblib import dump
import json

b_usr = "../../cluster_result/"
values = ["silhouette", "calinski_harabasz", "davies_bouldin"]


def run_clustering(dataset, features, title, base_url):
    """
    Executes clustering using KMeans, ISOMAP, and t-SNE on the input dataset and features.
    Visualizes results, saves models, and logs best parameters and scores.
    """

    # KMeans Clustering
    try:
        best_params_k, best_score_k = define_best_params_kmeans(features)
        save_best_score(
            best_score_k, f"{title}_kmeans", f"{b_usr}{base_url}_kmeans/")
        for score_type in values:
            best_params_kmeans = best_params_k[score_type]
            kmeans = KMeans(n_clusters=best_params_kmeans)
            kmeans_labels = kmeans.fit_predict(features)
            # Visualize, save models and results for KMeans
            handle_clustering_results(dataset, features, kmeans, None,
                                      best_params_kmeans, "kmeans", title, score_type, base_url)
    except Exception as e:
        print(f"Something went wrong in KMeans: {e}")

    # ISOMAP Clustering
    try:
        best_params_i, best_score_i = define_best_params_isomap(features)
        save_best_score(
            best_score_i, f"{title}_isomap", f"{b_usr}{base_url}_isomap/")
        for score_type in values:
            best_params_isomap = best_params_i[score_type]
            isomap, isomap_kmeans = best_isomap_kmeans(best_params_isomap)
            isomap_projection = isomap.fit_transform(features)
            isomap_labels = isomap_kmeans.fit_predict(isomap_projection)
            # Visualize, save models and results for ISOMAP
            handle_clustering_results(dataset, features, isomap_kmeans, isomap,
                                      best_params_isomap, "isomap", title, score_type, base_url)
    except Exception as e:
        print(f"Something went wrong in Isomap-KMeans: {e}")

    # t-SNE Clustering
    try:
        best_params_t, best_score_t = define_best_params_tsne(features)
        save_best_score(
            best_score_t, f"{title}_tsne", f"{b_usr}{base_url}_tsne/")
        for score_type in values:
            best_params_tsne = best_params_t[score_type]
            tsne, tsne_kmeans = best_tsne_kmeans(best_params_tsne)
            tsne_projection = tsne.fit_transform(features)
            tsne_labels = tsne_kmeans.fit_predict(tsne_projection)
            # Visualize, save models and results for t-SNE
            handle_clustering_results(
                dataset, features, tsne_kmeans, tsne, best_params_tsne, "tsne", title, score_type, base_url)
    except Exception as e:
        print(f"Something went wrong in TSNE-KMeans: {e}")


def handle_clustering_results(dataset, features, model, transformer, best_params, method, title, score_type, base_url):
    """
    Helper function to handle the visualization, saving of models, and logging of results for clustering.
    """
    plot_results(transformer, model, best_params, method, features,
                 f"{title}_{method}_{score_type}", f"{b_usr}{base_url}_{method}/")
    plot_images(dataset, model.labels_,
                f"{title}_{method}_{score_type}", f"{b_usr}{base_url}_{method}/")
    save_model(model, f"{title}_{method}_{score_type}",
               f"{b_usr}{base_url}_{method}/")
    if transformer:
        save_model(
            transformer, f"{title}_{method}_{score_type}", f"{b_usr}{base_url}_{method}/")
    save_best_params(
        best_params, f"{title}_{method}_{score_type}", f"{b_usr}{base_url}_{method}/")
    print(f"Best params {method}: {best_params}")


def save_model(model, name, url):
    """Saves the trained model to the specified path."""
    dump(model, f'{url}{name}.joblib')


def save_best_params(best_params, name, url):
    """Saves the best parameters used for clustering to a JSON file."""
    with open(f'{url}best_params_{name}.json', 'w') as f:
        json.dump(best_params, f)


def save_best_score(best_score, name, url):
    """Saves the best scores obtained from clustering to a JSON file."""
    with open(f'{url}best_score_{name}.json', 'w') as f:
        json.dump(best_score, f)
