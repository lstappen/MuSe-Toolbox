import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import diarisation.radar_chart as radar_chart
from diarisation.utils import get_cluster_data_points, remove_feature_suffix, concat_data_and_labels, scale_data


def get_colors(n_clusters):
    # returns list of n_clusters + 1 colors (last color is for 'all data')
    cmap = plt.get_cmap('nipy_spectral')(np.linspace(0, 1, n_clusters + 2))
    return cmap[1:]


def plot_most_distinctive_features_per_cluster(data, labels, emo_dim, export_dir=None, show=False, standardize=False,
                                               format='png'):
    emos_long = ' + '.join([x.capitalize() for x in emo_dim])
    emos_short = '+'.join([x[0].upper() for x in emo_dim])

    if standardize:
        data = scale_data(data)

    features_mean_total = data.mean(axis=0)
    data_labels = concat_data_and_labels(data, labels)
    colors = get_colors(len(np.unique(labels)))
    for i in np.unique(labels):
        cluster_data = get_cluster_data_points(data_labels, i)
        features_mean_cluster = cluster_data.mean(axis=0)
        diff = features_mean_cluster - features_mean_total
        chart_labels = diff.abs().nlargest(min(8, diff.size)).index.values

        chart = radar_chart.RadarChart(chart_labels)
        chart_data_total = features_mean_total[chart_labels].tolist()
        chart.add_to_radar("All data", colors[-1], chart_data_total)
        chart_data_cluster = features_mean_cluster[chart_labels].tolist()
        chart.add_to_radar(f"${emos_short}_{i}$", colors[i], chart_data_cluster)
        standardized_str = ' (standardised)' if standardize else ''
        chart_title = f"{emos_long} - cluster {i}: most distinctive features{standardized_str}"
        filename = f"distinctive_features_cluster_{i}{standardized_str}"
        chart.show(chart_title, show, export_dir, filename, format)

    return


def plot_most_distinctive_features_over_all_clusters(data, labels, emo_dim, export_dir=None, show=False,
                                                     standardize=False, format='png'):
    emos_long = ' + '.join([x.capitalize() for x in emo_dim])
    emos_short = '+'.join([x[0].upper() for x in emo_dim])

    if standardize:
        data = scale_data(data)

    features_mean_total = data.mean(axis=0)
    data_labels = concat_data_and_labels(data, labels)
    colors = get_colors(len(np.unique(labels)))
    counter = pd.Series(0, features_mean_total.index)
    features_mean_clusters = []

    df = features_mean_total.copy().to_frame('all_data_mean')
    features_std_total = data.std(axis=0)
    df = pd.concat([df, features_std_total.to_frame('all_data_std')], sort=False, axis=1)

    for i in np.unique(labels):
        cluster_data = get_cluster_data_points(data_labels, i)
        features_mean_cluster = cluster_data.mean(axis=0)
        diff = features_mean_cluster - features_mean_total
        features_mean_clusters.append(features_mean_cluster)

        df = pd.concat([df, features_mean_cluster.to_frame(f"{i}_mean")], sort=False, axis=1)
        features_std_cluster = cluster_data.std(axis=0)
        df = pd.concat([df, features_std_cluster.to_frame(f"{i}_std")], sort=False, axis=1)

        counter += diff.abs()

    chart_labels = counter.nlargest(min(8, counter.size)).index.values
    chart = radar_chart.RadarChart(chart_labels)
    chart_data_total = features_mean_total[chart_labels].tolist()
    chart.add_to_radar("All data", colors[-1], chart_data_total)
    for i in np.unique(labels):
        chart_data_cluster = features_mean_clusters[i][chart_labels].tolist()
        chart.add_to_radar(f"${emos_short}_{i}$", colors[i], chart_data_cluster)

    standardized_str = ' (standardised)' if standardize else ''
    chart_title = f"Most distinctive features for {emos_long}{standardized_str}"
    standardized_str = '_standardised' if standardize else ''
    filename = f"distinctive_features_all_clusters{standardized_str}"
    chart.show(chart_title, show, export_dir, filename, format)
    if export_dir is not None:
        df.to_csv(f"{export_dir}/features_per_cluster{standardized_str}.csv")
    return


def plot_target_correlation(data, target, export_dir=None, show=False, absolute=True, format='png'):
    if isinstance(target, pd.DataFrame):
        target = target.iloc[:, 0]
    target_name = target.name
    if target_name in data.columns.values:
        data = data.drop(columns=[target_name])
    data_with_target = concat_data_and_labels(data, target)

    corr_matrix = data_with_target.corr().round(2)
    corr_target = corr_matrix[target_name].drop(target_name)
    if absolute:
        corr_target = corr_target.abs()

    w, h = 10, 1 + 0.2 * data.shape[1]
    fig = corr_target.nlargest(data.shape[1]).plot(kind='barh', figsize=(w, h)).get_figure()
    abs_str = 'Absolute ' if absolute else ''
    plt.title(f'{abs_str}Correlation between {target_name} and all other features')
    plt.tight_layout()

    if show:
        plt.show()
    if export_dir is not None:
        abs_str = '_abs' if absolute else ''
        fig.savefig(f"{export_dir}/correlation_{target_name}_features{abs_str}.{format}", format=format)
    plt.close()


def plot_clusters(data, labels, feature_y='mean_arousal', feature_x='mean_valence', export_dir=None, show=False,
                  format='png'):
    """
    Plot the resulting clusters with regard to specific features.
    """
    n_clusters = len(np.unique(labels))
    data_labels = concat_data_and_labels(data, labels)

    colors = get_colors(n_clusters)
    fig, axes = plt.subplots(nrows=1, ncols=1)
    for i in np.unique(labels):
        cluster_data = get_cluster_data_points(data_labels, i, False)
        axes.scatter(cluster_data.loc[:, feature_x], cluster_data.loc[:, feature_y], marker='.', s=30, lw=0, alpha=0.5,
                     color=colors[i], edgecolor='k', label=i)
    axes.set_title("Visualisation of the clustered data.")
    axes.set_xlabel(feature_x)
    axes.set_ylabel(feature_y)
    axes.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    if show:
        plt.show()
    # save the figure
    if export_dir is not None:
        if remove_feature_suffix(feature_y) == remove_feature_suffix(feature_x):
            fig_name = remove_feature_suffix(feature_y)
        else:
            fig_name = f"{feature_y},{feature_x}"
        fig.savefig(f"{export_dir}/clustered_data_visualisation({fig_name}).{format}", format=format)
    plt.close()


def plot_histogram(data_valence, data_arousal, format='svg'):
    """Generate a histogram for mean valence and mean arousal.

    Args:
        data_valence (pd.DataFrame): mean valence data
        data_arousal (pd.DataFrame): mean arousal data
    """
    bins = 20
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))
    data_arousal.hist(density=1, histtype="step", bins=bins, ax=axes[0])
    data_valence.hist(density=1, histtype="step", bins=bins, ax=axes[1])
    axes[0].set_title('Arousal segments mean')
    axes[1].set_title('Valence segments mean')
    # fig.suptitle("Histogram segments")
    plt.show()
    fig.savefig(f"histogram_segments.{format}", format=format)


def plot_boxplot(data_valence, data_arousal, format='svg'):
    """Boxplot for mean valence and mean arousal

    Args:
        data_valence (pd.DataFrame): mean valence data
        data_arousal (pd.DataFrame): mean arousal data
    """
    if isinstance(data_valence, pd.Series):
        data_valence = data_valence.to_frame()
    if isinstance(data_arousal, pd.Series):
        data_arousal = data_arousal.to_frame()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 4))
    data_arousal.boxplot(ax=axes[0])
    data_valence.boxplot(ax=axes[1])
    axes[0].set_title('Arousal')
    axes[1].set_title('Valence')
    # fig.suptitle("Box-plot for segments")
    plt.show()
    fig.savefig(f"boxplot_segments.{format}", format=format)
