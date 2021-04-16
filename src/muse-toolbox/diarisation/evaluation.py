import os

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from s_dbw import S_Dbw

from diarisation.utils import concat_data_and_labels


class Evaluation:
    """
        Evaluation of the clustering results with internal validation measures. 
        Internal clustering validation measures were selected based on the paper:
        Understanding of Internal Clustering Validation Measures, DOI: 10.1109/icdm.2010.35
        The used measures are: calinski_harabasz_index, silhouette_index, davies_brouldin_index and s_dbw_validity_index
    """

    def __init__(self, data: pd.DataFrame, cluster_labels, fpc, remove_noise=False, export_dir=''):
        self.data = data
        self.labels = cluster_labels  # labels for S_dbw has to be numpy array
        # self.cluster_labels = cluster_labels
        self.cluster_labels = pd.DataFrame(cluster_labels, columns=["labels"])
        # Reset the index -> else concat will produce NAN values
        self.data.reset_index(drop=True, inplace=True)
        self.cluster_labels.reset_index(drop=True, inplace=True)
        if remove_noise:
            self.data, self.cluster_labels, self.labels = self.remove_noise()
        # self.labels = pd.DataFrame(cluster_labels, columns=["labels"])
        # self.data_labels = pd.concat([self.data, self.labels], sort=False, axis=1)
        self.data_labels = concat_data_and_labels(self.data, self.cluster_labels)
        self.number_clusters = len(np.unique(self.cluster_labels))
        self.export_dir = export_dir
        self.save_outputs = export_dir is not None and export_dir != ""

        # Validation measures
        self.ch = 0
        self.s = 0
        self.db = 0
        self.s_dbw = 0
        self.fpc = fpc

    def remove_noise(self):
        """Removes the data points that have a label of -1.
        This label is only produced by DBSCAN

        Returns:
            pd.DataFrame, np.array, pd.DataFrame: the cleaned dataFrame, the cleaned labels for s_dbw, 
            and the cleaned labels for the other validation measures
        """
        # Remove noise from data:
        data = concat_data_and_labels(self.data, self.cluster_labels)
        condition = data["labels"] != -1
        data = data[condition]

        data = data.dropna()

        # Remove noise from labels
        # condition = labels["labels"] != -1
        labels = data.filter(["labels"])
        cluster_labels = labels.to_numpy()
        cluster_labels = cluster_labels.reshape([len(labels), ])

        # cast the labels to ints
        cluster_labels = cluster_labels.astype(int)
        data = data.drop(["labels"], axis=1)
        return data, labels, cluster_labels

    def calinski_harabasz_index(self):
        """Calculates the calinski_harabasz_index

        Returns:
            [float]: The calinski_harabasz_index
        """
        self.ch = metrics.calinski_harabasz_score(self.data, self.labels)
        return self.ch

    def silhouette_index(self):
        """Calculates the silhouette_index

        Returns:
            [float]: The silhouette_index
        """
        self.s = metrics.silhouette_score(self.data, self.labels)
        return self.s

    def davies_brouldin_index(self):
        """Calculates the davies_brouldin_index

        Returns:
            [float]: The davies_brouldin_index
        """
        self.db = metrics.davies_bouldin_score(self.data, self.labels)
        return self.db

    def s_dbw_validity_index(self):
        """Calculates the s_dbw_validity_index

        Returns:
            [float]: The s_dbw_validity_index
        """
        data = self.data.to_numpy()
        self.s_dbw = S_Dbw(data, self.labels, centers_id=None, method='Halkidi', alg_noise='bind', centr='mean',
                           nearest_centr=True, metric='euclidean')
        return self.s_dbw

    def get_results(self):
        """
        Gathers the results of the evaluation measures

        Returns:
            pd.DataFrame: A DataFrame that contains all evaluation measures.
        """
        if self.save_outputs:
            self.data_labels.to_csv(f"{self.export_dir}/clustering_results.csv")

        self.calinski_harabasz_index()
        self.silhouette_index()
        self.davies_brouldin_index()
        self.s_dbw_validity_index()
        values = np.array([[self.number_clusters, self.ch, self.s, self.db, self.s_dbw, self.fpc]])
        results = pd.DataFrame(values, columns=["Number of clusters", "CH", "S", "DB", "S_dbw", "FPC"])
        return results

    def export_results_as_labels(self, export, nan_indices, output_path, seg_info_path):
        """
        Exports labels in the same format as the original labels (one file per video)

        Args:
            export (str): name of the label set (export folder will be named after this)
            nan_indices (list): indices of rows in input data that have been dropped because of NaN values
            output_path (str): output directory where the export folder will be stored
            seg_info_path (str): path to segment info files; used as reference for label files
        """

        print("Exporting results as labels...")
        src = seg_info_path

        dest = os.path.join(output_path, export, export)
        if not os.path.exists(dest):
            os.makedirs(dest)

        label_base_files = [int(f.split('.')[0]) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        label_base_files = sorted(label_base_files)

        # putting nan-values back in the correct place
        segment_labels = self.data_labels['labels'].values
        full_length = segment_labels.shape[0] + len(nan_indices)
        segment_labels_full = np.zeros(full_length, dtype=int)
        seg_idx = 0
        for i in range(full_length):
            if i in nan_indices:
                segment_labels_full[i] = -1
            else:
                segment_labels_full[i] = segment_labels[seg_idx]
                seg_idx += 1

        start = 0

        for f in label_base_files:
            df = pd.read_csv(os.path.join(src, f"{f}.csv"))
            end = start + len(df.index)
            # print(f"{f}: Start: {start}, End: {end}")

            df['class_id'] = segment_labels_full[start:end]
            df.loc[df['class_id'] == -1, 'class_id'] = pd.NA

            df.to_csv(os.path.join(dest, f"{f}.csv"), index=False)
            start = end
        print(f"Labels filled: {start} of {segment_labels_full.shape[0]}")
