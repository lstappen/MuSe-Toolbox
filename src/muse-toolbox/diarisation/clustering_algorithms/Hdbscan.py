import hdbscan


def run_hdbscan(data, metric='euclidean', min_cluster_size=5, min_sampels=None, eps=0.0):
    """Apply HDBSCAN on a dataset
    For more see: https://hdbscan.readthedocs.io/en/latest/

    Args:
        data (pd.DataFrame): The dataset
        metric (str, optional): The distance metirc. Defaults to 'euclidean'.
        min_cluster_size (int, optional): The minimum size of clusters. Defaults to 5.
        min_sampels ([type], optional): The number of samples in a neighbourhood for a point to be considered a core point.. Defaults to None.
        eps (float, optional): A distance threshold. Clusters below this value will be merged. Defaults to 0.0.

    Returns:
         ndarray, ndarray, ndarray: Cluster labels for each point, number_of_clusters,
        and the strength with which each sample is a member of its assigned cluster.
    """
    if min_sampels == 0:
        min_sampels = None

    clusterer = hdbscan.HDBSCAN(metric=metric, min_cluster_size=min_cluster_size, min_samples=min_sampels,
                                cluster_selection_epsilon=0)
    clusterer = clusterer.fit(data)
    return clusterer.labels_, clusterer.labels_.max(), clusterer.probabilities_
