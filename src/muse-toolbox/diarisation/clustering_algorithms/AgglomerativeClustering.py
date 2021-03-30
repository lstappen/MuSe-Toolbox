from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


def run_agglomerative_clustering(data, n_clusters=None, metric="euclidean", linkage="ward", distance_treshold=None):
    """ Apply Agglomerative hierarchial clustering on a dataset. 
    For more see: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering

    Args:
        data (pd.DataFrame): The used dataset
        n_clusters (int, optional): The number of clusters as stopping criterion. Defaults to None.
        metric (str, optional): The distance metric. Defaults to "euclidean".
        linkage (str, optional): The linkage method. Defaults to "ward".
        distance_treshold ([type], optional): [The linkage distance threshold above which, 
        clusters will not be merged. Defaults to None.

    Returns:
        [type]: [description]
    """
    clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity=metric, linkage=linkage,
                                         distance_threshold=distance_treshold).fit(data)
    return clustering.n_clusters_, clustering.labels_


def dendrogram_plot(data):
    """ Plot the dendrogram for a given dataset

    Args:
        data (pd.DataFrame): The used dataset
    """
    cluster_linkage_array = linkage(data, 'ward')
    fig = plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    dendrogram(cluster_linkage_array, p=8, orientation='top', distance_sort='descending',
               show_leaf_counts=True, show_contracted=True)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.show()
    fig.savefig("Dendrogram.svg", format="svg")
