from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Define possible numbers of clusters to evaluate
parameters = {'n_clusters': [5, 6, 7, 10]}


def define_best_params_kmeans(dataset):
    """
    Function to determine the best number of clusters for KMeans clustering 
    by evaluating various cluster validity indices.
    """
    results = []

    # Iterate over different numbers of clusters
    for param in parameters['n_clusters']:
        # Initialize and fit KMeans model
        kmeans = KMeans(n_clusters=param)
        kmeans.fit(dataset)

        # Compute clustering performance metrics
        sse = kmeans.inertia_
        silhouette_avg = silhouette_score(dataset, kmeans.labels_)
        calinski_harabasz = calinski_harabasz_score(dataset, kmeans.labels_)
        davies_bouldin = davies_bouldin_score(dataset, kmeans.labels_)

        # Append results to the list
        results.append({
            'params': param,
            'sse': sse,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin
        })

    # Determine the best number of clusters based on different metrics
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
