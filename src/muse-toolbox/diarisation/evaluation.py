import os

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from s_dbw import S_Dbw

import diarisation.config as cfg
from diarisation.utils import concat_data_and_labels


class Evaluation:
    """
        Evaluation of the clustering results with internal validation measures. 
        Internal clustering validation measures were selected based on the paper:
        Understanding of Internal Clustering Validation Measures, DOI: 10.1109/icdm.2010.35
        The used measures are: calinski_harabasz_index, silhouette_index, davies_brouldin_index and s_dbw_validity_index
    """

    def __init__(self, data: pd.DataFrame, cluster_labels, fpc, plot=False, remove_noise=False, export_dir=''):
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
        self.plot = plot
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
        This label is only produced by DBSCAN or HDBSCAN

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

    '''# Plot the first two entries
    def show_plot(self, feature_x='', feature_y=''):
        """
        Plot the resulting clusters.
        For the y-axis use the first column corresponding to the mean_arousal, 
        and for the x-axis use the second column corresponding to the mean_valence.  
        """
        colors = cm.nipy_spectral(np.linspace(0, 1, self.number_clusters))
        fig, axes = plt.subplots(nrows=1, ncols=1)
        for i in np.unique(self.cluster_labels):
            cluster_data = cluster_naming.get_cluster_data_points(self.data_labels, i)
            # axes.scatter(cluster_data["mean_valence"], cluster_data["mean_arousal"], marker='.', s=30, lw=0,
            # alpha=0.5, color=colors[i], edgecolor='k', label=i)
            # valence 1 arousal 0
            axes.scatter(cluster_data.iloc[:, 1], cluster_data.iloc[:, 0], marker='.', s=30, lw=0, alpha=0.5,
                         color=colors[i], edgecolor='k', label=i)
        axes.set_title("The visualization of the clustered data.")
        axes.set_xlabel(self.data_labels.columns.values[1])
        axes.set_ylabel(self.data_labels.columns.values[0])
        axes.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
        axes.set_ylim(-1, 1)
        axes.set_xlim(-1, 1)
        plt.show()
        # save the figure
        if self.save_outputs:
            fig.savefig(f"{self.export_dir}/clustered_data_visualisation.png", format="png")
    '''

    def get_results(self):
        """
        Gathers the results of the evaluation measures

        Returns:
            pd.DataFrame: A DataFrame that contains all evaluation measures.
        """
        if self.save_outputs:
            self.data_labels.to_csv(f"{self.export_dir}/clustering_results.csv")
        #if self.plot:
        #    self.show_plot()

        self.calinski_harabasz_index()
        self.silhouette_index()
        self.davies_brouldin_index()
        self.s_dbw_validity_index()
        values = np.array([[self.number_clusters, self.ch, self.s, self.db, self.s_dbw, self.fpc]])
        results = pd.DataFrame(values, columns=["Number of clusters", "CH", "S", "DB", "S_dbw", "FPC"])
        return results

    def export_results_as_labels(self, export, nan_indices, seg_type):
        """
        Exports labels in the same format as the original labels (one file per video)

        Args:
            export (str): name of the label set (export folder will be named after this)
            nan_indices (list): indices of rows in input data that have been dropped because of NaN values
            seg_type (str): segment type (topic or wild)
        """
        # TODO move function outside of Evaluation
        src = cfg.LABEL_BASE_PATH + seg_type

        dest = os.path.join(cfg.LABEL_EXPORT_PATH, export)
        if not os.path.exists(dest):
            os.makedirs(dest)

        label_base_files = [int(f.split('.')[0]) for f in os.listdir(src) if os.path.isfile(os.path.join(src, f))]
        label_base_files = sorted(label_base_files)

        # putting nan-values back in the correct place
        segment_labels = self.data_labels['labels'].values
        print(f"Segment labels shape(without nan): {segment_labels.shape}")
        print(f"nan-indices: {nan_indices}")
        # segment_labels_full = np.insert(segment_labels, nan_indices, -1, 0)  # can't fill with np.nan because type is int
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

            df.loc[:, 'class_id'] = segment_labels_full[start:end]
            df.loc[df['class_id'] == -1, 'class_id'] = pd.NA  # or np.nan

            df.to_csv(os.path.join(dest, f"{f}.csv"), index=False)
            start = end
        print(f"Labels filled: {start} of {segment_labels_full.shape[0]}")
