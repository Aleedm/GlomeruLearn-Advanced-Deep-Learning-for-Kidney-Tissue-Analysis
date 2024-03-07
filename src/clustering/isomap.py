from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Define a parameter grid for Isomap and KMeans
param_grid = {
    'isomap__n_components': [2, 3, 5, 10, 30],
    'isomap__n_neighbors': [5, 10, 20, 50, 100],
    'kmeans__n_clusters': [5, 7, 10]
}

# Set up a pipeline with Isomap for dimensionality reduction and KMeans for clustering
pipe = Pipeline(steps=[
    ('isomap', Isomap()),
    ('kmeans', KMeans())
])


def define_best_params_isomap(dataset):
    """
    Determine the best parameters for Isomap and KMeans by evaluating different combinations
    and calculating several cluster validity indices.
    """
    results = []

    # Iterate over the parameter grid and set the parameters for the pipeline
    for params in ParameterGrid(param_grid):
        pipe.set_params(**params)
        pipe.fit(dataset)

        # Compute the Sum of Squared Errors (SSE) and clustering metrics
        sse = pipe['kmeans'].inertia_
        silhouette_avg = silhouette_score(dataset, pipe['kmeans'].labels_)
        calinski_harabasz = calinski_harabasz_score(
            dataset, pipe['kmeans'].labels_)
        davies_bouldin = davies_bouldin_score(dataset, pipe['kmeans'].labels_)

        # Append the results including parameters and scores
        results.append({
            'params': params,
            'sse': sse,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin
        })

    # Find the best parameters based on different scores
    best_sse_result = min(results, key=lambda x: x['sse'])
    best_silhouette_result = max(results, key=lambda x: x['silhouette_score'])
    best_calinski_harabasz_result = max(
        results, key=lambda x: x['calinski_harabasz_score'])
    best_davies_bouldin_result = min(
        results, key=lambda x: x['davies_bouldin_score'])

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


def best_isomap_kmeans(best_params):
    """
    Extract the best parameters for Isomap and KMeans and create instances with these parameters.
    """
    isomap_params = {k.replace('isomap__', ''): v for k,
                     v in best_params.items() if k.startswith('isomap__')}
    isomap = Isomap(**isomap_params)

    kmeans_params = {k.replace('kmeans__', ''): v for k,
                     v in best_params.items() if k.startswith('kmeans__')}
    kmeans = KMeans(**kmeans_params)
    return isomap, kmeans
