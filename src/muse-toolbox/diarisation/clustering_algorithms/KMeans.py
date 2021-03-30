from sklearn.cluster import KMeans


# Todo: Write docstrings


def run_kmeans(data_to_fit, data_to_predict, number_of_clusters, seed=1):
    """ Apply the k-means algorithm on a given dataset.
    For more see: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    Args:
        data_to_fit ([type]): The data set to be clustered
        data_to_predict ([type]): The data set to be predicted
        number_of_clusters (int): The number of clusters
        seed (int, optional): random seed for kmeans initialisation (default: 1)

    Returns:
        KMeans, [int]: Kmeans object and the cluster id for each data point
    """
    km = KMeans(n_clusters=number_of_clusters, init='k-means++', max_iter=5000, random_state=seed)
    labels = km.fit_predict(data_to_fit)
    if data_to_predict is None:
        preds = None
    else:
        preds = km.predict(data_to_predict)
    return km, labels, preds
