from sklearn.cluster import DBSCAN


def run_dbscan(data, metric='euclidean', eps=0.5, min_samples=5):
    """ Applies DBSCAN on a dataset. 
    For more see: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

    Args:
        data (pd.DataFrame): The used dataset
        metric (str, optional): The distance metric. Defaults to 'euclidean'.
        eps (float, optional): The maximum distance between two samples 
        for one to be considered as in the neighborhood of the other. Defaults to 0.5.
        min_samples (int, optional): The number of samples (or total weight) in a 
        neighborhood for a point to be considered as a core point. Defaults to 5.

    Returns:
        [type]: [description]
    """
    clustering = DBSCAN(metric=metric, eps=eps, min_samples=min_samples).fit(data)
    return clustering.labels_, clustering.core_sample_indices_
