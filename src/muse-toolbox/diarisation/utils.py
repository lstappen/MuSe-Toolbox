import pandas as pd
import numpy as np
import os
import sys
import itertools as iter

import diarisation.config as cfg
from diarisation.clustering_algorithms import AgglomerativeClustering, FuzzyCMeans, GaussianMixturModels, KMeans, \
    Dbscan


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.log.flush()


# Load Data
def load_data(emo_dims, seg_type, use_standardised=False):
    """ Loads the arousal and valence data

    Args:
        emo_dims (list): List of emotional dimensions to use (e.g. arousal, valence)
        seg_type (string): Which segments to load (wild or topic)
        use_standardised (bool): whether to load standardised version ('standardised'-arg from cli)

    Returns:
        list(pd.DataFrame):  list of pandas DataFrames for each emotional dimension
    """
    segments = []
    for emo_dim in emo_dims:
        filler = "standardised_" if use_standardised else ''
        segments_file = os.path.join(cfg.DATA_PATH, f"{emo_dim}_{seg_type}_{filler}features.csv")
        segments.append(pd.read_csv(segments_file, header=0))
    return segments


def select_features(features, segments, emo_dims):
    """ Selects the appropriate features for arousal and valence, removes NAN values and concat the dataframes.
    Returns:
        pd.DataFrame: DataFrame with the selected features for arousal and valence
    """
    data = pd.concat(segments, sort=False, axis=1)
    nan_mask = data.isnull().any(axis=1)
    nan_indices = np.where(nan_mask)[0]
    print(f"Length data: {len(data.index)}")
    data = data.dropna()
    print(f"Length after dropna: {len(data.index)}")
    data_mean = data[[f'mean_{emo_dim}' for emo_dim in emo_dims]].copy()
    partitions = data[cfg.PARTITION_KEY]
    data = data[features]
    return data, data_mean, list(nan_indices), partitions


def extract_features_cli(features_cli, available_features=[], emo_dims=['arousal', 'valence']):
    """Creates a list of arousal and valence features according to the user input.

    Args:
        features_cli (list): The selected features from the cli.
        available_features (list or list-like): List of available features (both arousal and valence). Defaults to [].
        emo_dims (list): List of emotional dimensions to use (e.g. arousal, valence)

    Returns:
        [string]: List of the selected features for both arousal and valence
    """
    available_features = list(available_features)

    # replace feature set keywords with corresponding list of features
    cli_feature_sets = list(set(features_cli) & set(cfg.FEATURE_SETS))
    features_cli = [cfg.FEATURE_SETS[feat] if feat in cli_feature_sets else [feat] for feat in features_cli]
    features_cli = list(iter.chain.from_iterable(features_cli))

    features = []
    if 'all' in features_cli:
        features_to_ignore = [cfg.VIDEO_ID_KEY, cfg.SEGMENT_ID_KEY] + cfg.METADATA_KEYS
        features_to_ignore += ['start', 'end', 'id_arousal', 'id_valence']  # needed for old features
        features += [f for f in available_features if f not in features_to_ignore]
    else:
        for feat in features_cli:
            if feat in available_features:
                features.append(feat)
            else:
                for emo_dim in emo_dims:
                    if feat + f"_{emo_dim}" in available_features:
                        features.append(feat + f"_{emo_dim}")
    print(f'Features used: {features}')
    return features


def apply_clustering(data, params, partitions):
    labels = []
    fpc = 0

    if params.partitions[0] == 'all':
        data_to_fit = data
        data_to_predict = None
    else:
        partitions.reset_index(drop=True, inplace=True)
        data.reset_index(drop=True, inplace=True)
        data_to_fit = data.loc[partitions.isin(params.partitions)]
        data_to_predict = data.loc[~partitions.isin(params.partitions)]

    # K-means
    if params.kmeans:
        km, labels, preds = KMeans.run_kmeans(data_to_fit, data_to_predict, params.k, params.cluster_seed)

    # Fuzzy-C-Means:
    if params.fuzzyCMeans:
        labels, preds, fpc = FuzzyCMeans.run_fuzzy_cmeans(data_to_fit, data_to_predict, params.k, params.m,
                                                          params.cluster_seed)

    # DBSCAN
    if params.dbscan:
        # TODO cluster partitions
        labels, _ = Dbscan.run_dbscan(data=data, eps=params.eps, min_samples=params.min_samples)

    # Agglomerative Clustering
    if params.aggl:
        # Todo: add a lable to disable the plot
        # TODO cluster partitions
        # AgglomerativeClustering.dendrogram_plot(data)
        if params.distance_thr is not None:
            _, labels = AgglomerativeClustering.run_agglomerative_clustering(data=data, n_clusters=None,
                                                                             distance_treshold=params.distance_thr)
        else:
            _, labels = AgglomerativeClustering.run_agglomerative_clustering(data=data, n_clusters=params.k,
                                                                             distance_treshold=None)
    # Gaussian Mixture Model
    if params.gmm:
        labels, preds = GaussianMixturModels.run_GMM(data_to_fit, data_to_predict, params.k, params.cluster_seed)

    if params.partitions[0] == 'all':
        merged_labels = labels
    else:
        # merge labels and preds
        merged_labels = []
        count_label = 0
        count_pred = 0
        for part in partitions.values:
            if part in params.partitions:
                merged_labels.append(labels[count_label])
                count_label += 1
            else:
                merged_labels.append(preds[count_pred])
                count_pred += 1
        merged_labels = np.array(merged_labels)

    return merged_labels, fpc


def concat_data_and_labels(data, labels):
    data.reset_index(drop=True, inplace=True)
    labels.reset_index(drop=True, inplace=True)
    data_labels = pd.concat([data, labels], sort=False, axis=1)
    return data_labels


def get_cluster_data_points(data, cluster, drop_labels=True):
    condition = data["labels"] == cluster
    data = data[condition]
    if drop_labels:
        data = data.drop(["labels"], axis=1)
    return data


def remove_feature_suffix(feature_name):
    """
    Removes suffixes '_arousal' and '_valence' from the feature name
    """
    suffixes = ['_arousal', '_valence']
    for suffix in suffixes:
        if feature_name.endswith(suffix):
            return feature_name.replace(suffix, '')
    return feature_name


def export_to_csv(export_dir, name, settings: pd.DataFrame, results: pd.DataFrame, names: pd.DataFrame):
    """Exports the settings, the results and the cluster names to a CSV file.

    Args:
        export_dir (string): Directory in which to save csv file
        name (string): Name of csv file
        settings (pd.DataFrame): Settings to be exported
        results (pd.DataFrame): Results to be exported
        names (pd.DataFrame): Names to be exported
    """
    csv_name: str = f"{export_dir}/{name}.csv"
    settings.to_csv(csv_name, index=False)
    results.to_csv(csv_name, mode="a", index=False)
    names.to_csv(csv_name, mode="a", index=False)


def append_to_csv(name, settings: pd.DataFrame, results: pd.DataFrame):
    """Append the settings and results of an experiment to a given csv file.

    Args:
        name (string): Filename of the csv file
        settings (pd.DataFrame): Settings to be append
        results (pd.DataFrame): Results to be append
    """
    csv_name: str = name + ".csv"
    results = pd.concat([settings, results], axis=1, sort=False)
    if os.path.exists(csv_name):
        results.to_csv(csv_name, mode="a", header=False, index=False)
    else:
        results.to_csv(csv_name, index=False)
