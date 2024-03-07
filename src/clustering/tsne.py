from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Define parameter grids for t-SNE and KMeans
tsne_params = {
    'n_components': [2, 3],
    'perplexity': [30, 50, 80, 100],
    'early_exaggeration': [12, 24, 36],
    'learning_rate': [100, 200],
    'n_iter': [1000]
}

kmeans_params = {
    'n_clusters': [5, 7, 10],
}


def define_best_params_tsne(dataset):
    """
    Function to perform grid search over t-SNE and KMeans hyperparameters, 
    and evaluate clustering performance using various metrics.
    """
    tsne_results = []

    # Iterate over all combinations of t-SNE parameters
    for tsne_param in ParameterGrid(tsne_params):
        tsne = TSNE(**tsne_param)
        X_transformed = tsne.fit_transform(dataset)

        # Iterate over all combinations of KMeans parameters
        for kmeans_param in ParameterGrid(kmeans_params):
            kmeans = KMeans(**kmeans_param)
            kmeans.fit(X_transformed)

            # Compute clustering performance metrics
            sse = kmeans.inertia_
            silhouette_avg = silhouette_score(X_transformed, kmeans.labels_)
            calinski_harabasz = calinski_harabasz_score(
                dataset, kmeans.labels_)
            davies_bouldin = davies_bouldin_score(dataset, kmeans.labels_)

            # Append results to the list
            tsne_results.append({
                'params': {**tsne_param, **kmeans_param},
                'sse': sse,
                'silhouette_score': silhouette_avg,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin
            })

    # Determine the best parameter combinations based on different metrics
    best_sse_result = min(tsne_results, key=lambda x: x['sse'])
    best_silhouette_result = max(
        tsne_results, key=lambda x: x['silhouette_score'])
    best_calinski_harabasz_result = max(
        tsne_results, key=lambda x: x['calinski_harabasz_score'])
    best_davies_bouldin_result = min(
        tsne_results, key=lambda x: x['davies_bouldin_score'])

    # Store the best parameters and scores
    best_params = {
        "sse": best_sse_result['params'],
        "silhouette": best_silhouette_result['params'],
        "calinski_harabasz": best_calinski_harabasz_result['params'],
        "davies_bouldin": best_davies_bouldin_result['params']
    }

    best_scores = {
        "sse": float(best_sse_result['sse']),
        "silhouette": float(best_silhouette_result['silhouette_score']),
        "calinski_harabasz": float(best_calinski_harabasz_result['calinski_harabasz_score']),
        "davies_bouldin": float(best_davies_bouldin_result['davies_bouldin_score'])
    }
    return best_params, best_scores


def best_tsne_kmeans(best_params):
    """
    Function to create TSNE and KMeans instances with the best parameters.
    """
    tsne_param_names = ['n_components', 'perplexity',
                        'early_exaggeration', 'learning_rate', 'n_iter']
    tsne_params = {name: best_params[name] for name in tsne_param_names}
    tsne = TSNE(**tsne_params)

    kmeans_param_names = ['n_clusters']
    kmeans_params = {name: best_params[name] for name in kmeans_param_names}
    kmeans = KMeans(**kmeans_params)
    return tsne, kmeans
