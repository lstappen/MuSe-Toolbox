import matplotlib.cm as cm
import numpy as np
import pandas as pd

import diarisation.radar_chart as radar_chart
from diarisation.utils import get_cluster_data_points, concat_data_and_labels


class ClusterNaming:
    """Assigns a discrete emotion to each cluster based on the profiling"""

    def __init__(self, selected_features: pd.DataFrame, labels, emo_dims, show, export_dir):
        """Initializes the ClusterNaming.

        Args:
            selected_features (pd.DataFrame): The selected features
            labels ([int]): The cluster to which a points belongs
            emo_dims ([str]): Emotional dimensions used (e.g. arousal, valence)
            show (bool): If true show the RadarChart
            export_dir (str): If not empty, export the profiling to a csv file in export_dir
        """
        self.start = 0
        self.emo_dims = emo_dims
        self.show = show
        self.export_dir = export_dir
        self.export_csv = export_dir is not None and export_dir != ""
        if -1 in labels:
            self.start = -1
        self.selected_features = selected_features
        self.n_clusters = len(np.unique(labels))

        columns = ["cluster", "category", "name", "data_per_cluster"]
        for emo in emo_dims:
            columns += [f"mean_{emo}", f"median_{emo}", f"var_{emo}", f"std_{emo}", f"min_{emo}", f"max_{emo}",
                        f"percentile_5_{emo}", f"percentile_10_{emo}", f"percentile_25_{emo}", f"percentile_33_{emo}",
                        f"percentile_66_{emo}", f"percentile_75_{emo}", f"percentile_90_{emo}", f"percentile_95_{emo}"]
        self.attributes = pd.DataFrame(columns=columns)
        self.labels = pd.DataFrame(labels, columns=["labels"])
        self.n_rows = len(selected_features.index)

    # Watch out the first the columns are used to calculate the data
    def naming(self):
        """ Determines the discrete emotion and the category for each datapoint
        The category consists of a pair of valence and arousal

        Returns:
            (pd.DataFrame): a dataFrame with the discrete emotions and name for each data point
        """
        data = concat_data_and_labels(self.selected_features, self.labels)
        for i in range(self.start, self.n_clusters):
            data_cluster = get_cluster_data_points(data, i)
            self.calculate_attributes(data_cluster, i)
            if i == -1:
                # -1 represents noise -> only for DBSCAN and HDBSCAN
                self.attributes.loc[self.attributes.cluster == i, "category"] = "Noise"
                self.attributes.loc[self.attributes.cluster == i, "name"] = "Noise"
            else:
                category, name = self.set_name(i)
                self.attributes.loc[self.attributes.cluster == i, "category"] = category
                self.attributes.loc[self.attributes.cluster == i, "name"] = name
        if self.export_csv:
            self.export()
        return self.attributes

    def set_name(self, cluster):
        """Determines the name of a given cluster based on the profiling

        Args:
            cluster (int): Id of the cluster

        Returns:
            string, string: The first string is a combination of valence and arousal, 
            and the second string represents the discrete emotion category.
        """
        # load the data for a cluster 
        data_cluster = self.attributes[self.attributes["cluster"] == cluster]

        # arousal profiling
        if 'arousal' in self.emo_dims:
            percentile_33_arousal = data_cluster["percentile_33_arousal"].to_numpy()[0]
            percentile_66_arousal = data_cluster["percentile_66_arousal"].to_numpy()[0]

            if percentile_33_arousal >= 0.2:
                category_arousal = "high arousal"
            elif percentile_66_arousal <= -0.2:
                category_arousal = "low arousal"
            else:
                category_arousal = "medium arousal"
        else:
            category_arousal = "unknown arousal"

        # valence profiling
        if 'valence' in self.emo_dims:
            median_valence = data_cluster["median_valence"].to_numpy()[0]
            percentile_90_valence = data_cluster["percentile_90_valence"].to_numpy()[0]

            if percentile_90_valence <= 0.0:
                category_valence = "negative valence"
            else:
                category_valence = "positive valence"
        else:
            median_valence = -2
            category_valence = 'unknown valence'

        # naming
        if category_arousal == "high arousal" and category_valence == "negative valence":
            name = "anger"
        elif category_arousal == "high arousal" and category_valence == "positive valence":
            name = "joy"
        elif category_arousal == "low arousal" and category_valence == "negative valence":
            name = "sadness"
        elif category_arousal == "low arousal" and category_valence == "positive valence":
            name = "pleasure"
        elif category_arousal == "medium arousal" and (-0.2 <= median_valence <= 0.2):
            name = "neutral"
        elif category_arousal == "medium arousal" and category_valence == "positive valence":
            name = "joy / pleasure"
        elif category_arousal == "medium arousal" and category_valence == "negative valence":
            name = "anger / sadness"
        else:
            name = "unnamed"

        category = category_arousal + " " + category_valence

        return category, name

    def calculate_attributes(self, data: pd.DataFrame, cluster):
        """Calculate the following features for a cluster:
        mean, median, var, std, min, max, percentile: 5, 10, 25, 33, 66, 75, 90, 95
        For the calculation only the first two columns are used since this column stores the mean of both arousal and valence for all experiments. 
        Args:
            data (pd.DataFrame): All data points that belong to a cluster 
            cluster (int): The id of the cluster
        """
        data = data.loc[:, [f'mean_{emo}' for emo in self.emo_dims]]
        # calculate column wise
        mean = data.mean(axis=0)
        median = data.median(axis=0)
        var = data.var(axis=0)
        std = data.std(axis=0)
        min = data.min(axis=0)
        max = data.max(axis=0)
        q5 = data.quantile(0.05, axis=0)
        q10 = data.quantile(0.10, axis=0)
        q25 = data.quantile(0.25, axis=0)
        q33 = data.quantile(0.33, axis=0)
        q66 = data.quantile(0.66, axis=0)
        q75 = data.quantile(0.75, axis=0)
        q90 = data.quantile(0.90, axis=0)
        q95 = data.quantile(0.95, axis=0)

        data_per_cluster = np.round(((len(data.index) / self.n_rows) * 100), 2)
        attribute_dict = {"cluster": cluster, "data_per_cluster": data_per_cluster}
        for i, emo in enumerate(self.emo_dims):
            attribute_dict.update({f"mean_{emo}": mean[i], f"median_{emo}": median[i], f"var_{emo}": var[i],
                                   f"std_{emo}": std[i], f"min_{emo}": min[i], f"max_{emo}": max[i],
                                   f"percentile_5_{emo}": q5[i], f"percentile_10_{emo}": q10[i],
                                   f"percentile_25_{emo}": q25[i], f"percentile_33_{emo}": q33[i],
                                   f"percentile_66_{emo}": q66[i], f"percentile_75_{emo}": q75[i],
                                   f"percentile_90_{emo}": q90[i], f"percentile_95_{emo}": q95[i]})
        self.attributes = self.attributes.append(attribute_dict, ignore_index=True)

    def plot_attributes(self):
        """ 
        Adds a cluster to the RadarChart with the selected attributes:
        "median_valence", "median_arousal", "percentile_33_valence", "percentile_66_valence",
        "percentile_90_valence","percentile_33_arousal", "percentile_66_arousal", 
        "percentile_90_arousal"
        """
        # TODO overhaul
        return

        labels = ["median_valence", "median_arousal", "percentile_33_valence", "percentile_66_valence",
                  "percentile_90_valence",
                  "percentile_33_arousal", "percentile_66_arousal", "percentile_90_arousal"]
        # always use cm.nipy_spectral
        color = cm.nipy_spectral(np.linspace(0, 1, self.n_clusters))
        chart = radar_chart.RadarChart(labels)
        for i in range(self.start, self.n_clusters):
            data = self.attributes[self.attributes["cluster"] == i]
            name = "Cluster: {}".format(i)
            data = data.drop(
                ["cluster", "category", "name", "data_per_cluster", "mean_valence", "min_valence", "max_valence",
                 "percentile_10_valence", "percentile_75_valence", "var_valence", "std_valence", "var_arousal",
                 "min_arousal", "max_arousal",
                 "mean_arousal", "std_arousal", "percentile_5_valence", "percentile_25_valence",
                 "percentile_75_arousal", "percentile_95_valence",
                 "percentile_5_arousal", "percentile_10_arousal", "percentile_25_arousal", "percentile_95_arousal"],
                axis=1)
            data = data.to_numpy()
            data = np.reshape(data, len(labels))
            data = data.tolist()
            chart.add_to_radar(name, color[i], data)
        chart.show("", self.show, self.export_dir)

    def export(self):
        """
        Exports the profiling results to a csv file
        """
        self.attributes.to_csv(f"{self.export_dir}/Profiling.csv", index=False)
