import os
import shutil
import sys
from itertools import chain

import pandas as pd
from tabulate import tabulate

import diarisation.cluster_naming as cluster_naming
import diarisation.evaluation as evaluation
import diarisation.plots as plots
import diarisation.utils as ut
from diarisation.feature_reduction import apply_feature_reduction


def diarisation(args):

    if args.kmeans:
        args.algorithm = "kmeans"
    elif args.dbscan:
        args.algorithm = "dbscan"
    elif args.aggl:
        args.algorithm = "aggl"
    elif args.gmm:
        args.algorithm = "gmm"
    elif args.fuzzyCMeans:
        args.algorithm = "fuzzyCMeans"

    if args.export is not None and args.export == 'auto':
        args.export = generate_name_from_args(args)
    print(args.export)

    setup_export_folder_and_logger(args)
    print(f"Args: {args}")

    # Feature selection and reduction:
    segments = ut.load_data(args.input_path, args.emo_dims)
    available_features = list(chain.from_iterable([list(segments_emo.columns) for segments_emo in segments]))
    features = ut.extract_features_cli(args.features, available_features, args.emo_dims)
    print(f'Features used: {features}')
    data, data_mean, nan_indices, partitions = ut.select_features(features, segments, args.emo_dims)
    data_raw = data.copy()

    print(f"Data shape: {data.shape}")

    if args.standardised:
        data = ut.scale_data(data)

    # apply tsne, pca or som-based feature selection before
    if args.reduce_separately:
        data_emo_dims = []

        for emo_dim in args.emo_dims:
            data_emo_dim = data.loc[:, [x for x in features if emo_dim in x]]
            data_emo_dim = apply_feature_reduction(data_emo_dim, args)
            if args.tsne is not None or args.pca is not None:
                n_components = args.tsne if args.tsne is not None else args.pca
                data_emo_dim.columns = [f'c_{i}_{emo_dim}' for i in range(n_components)]
            data_emo_dims.append(data_emo_dim)

        data = pd.concat(data_emo_dims, axis=1)
    else:
        data = apply_feature_reduction(data, args)
        if args.tsne is not None or args.pca is not None:
            n_components = args.tsne if args.tsne is not None else args.pca
            data.columns = [f'c_{i}' for i in range(n_components)]
    if args.som:
        data_raw = data.copy()

    # Plot Hist and Box
    if args.plotHistBox and 'valence' in args.emo_dims and 'arousal' in args.emo_dims:
        plots.plot_histogram(data_mean[f"mean_valence"], data_mean[f"mean_arousal"])
        plots.plot_boxplot(data_mean[f"mean_valence"], data_mean[f"mean_arousal"])

    # Clustering
    labels, fpc = ut.apply_clustering(data, args, partitions)

    # Cluster Evaluation
    cluster_eval = evaluation.Evaluation(data=data, cluster_labels=labels, fpc=fpc, remove_noise=args.remove_noise,
                                         export_dir=args.export_dir)
    results = cluster_eval.get_results()

    print(tabulate(results, headers=list(results.columns.values), tablefmt="grid"))

    # Cluster Naming
    naming = cluster_naming.ClusterNaming(data_mean, labels, args.emo_dims, args.plot, args.export_dir)
    cluster_names = naming.naming()
    # naming.plot_attributes()
    print(tabulate(cluster_names, headers=list(cluster_names.columns.values), tablefmt="grid"))

    # apply minimum class threshold to only accept reasonable class distributions
    if args.min_class_thr is not None:
        smallest_class = cluster_names['data_per_cluster'].values.min()
        # smallest class should have at least args.min_class_thr of chance level
        rule_of_thumb_threshold = (args.min_class_thr * 100.0) / float(results['Number of clusters'].values[0])
        distr_is_ok = smallest_class >= rule_of_thumb_threshold
        print(args.export)
        print(f"Smallest class ({'not ' if not distr_is_ok else ''}OK): {smallest_class} %")
        if not distr_is_ok:
            print(f"Time to abort...")
            sys.stdout = sys.__stdout__
            shutil.rmtree(args.export_dir)
            return

    # Plots
    if 'none' not in args.plot:
        print("Plotting figures...")
    label_df = pd.DataFrame(labels, columns=["labels"])
    plot_all = 'all' in args.plot
    args.show_title = not args.plot_no_title

    if plot_all or 'corr' in args.plot:
        plots.plot_target_correlation(data_raw, label_df, args.export_dir, show=False, absolute=False,
                                      format=args.plot_format, show_title=args.show_title)
    if plot_all or 'corr_abs' in args.plot:
        plots.plot_target_correlation(data_raw, label_df, args.export_dir, show=False, absolute=True,
                                      format=args.plot_format, show_title=args.show_title)

    if plot_all or 'distinctive_features_single' in args.plot:
        plots.plot_most_distinctive_features_per_cluster(data_raw, label_df, args.emo_dims, export_dir=args.export_dir,
                                                         show=False, standardize=False, format=args.plot_format,
                                                         show_title=args.show_title)
        plots.plot_most_distinctive_features_per_cluster(data_raw, label_df, args.emo_dims, export_dir=args.export_dir,
                                                         show=False, standardize=False, format=args.plot_format,
                                                         show_title=args.show_title)

    if plot_all or 'distinctive_features_combined' in args.plot:
        plots.plot_most_distinctive_features_over_all_clusters(data_raw, label_df, args.emo_dims, show=False,
                                                               export_dir=args.export_dir, standardize=False,
                                                               format=args.plot_format, show_title=args.show_title)
        plots.plot_most_distinctive_features_over_all_clusters(data_raw, label_df, args.emo_dims, show=False,
                                                               export_dir=args.export_dir, standardize=True,
                                                               format=args.plot_format, show_title=args.show_title)

    if plot_all or 'point_clouds' in args.plot:
        if args.tsne is not None or args.pca is not None:  # additional plot with first two components
            plots.plot_clusters(data, label_df, data.columns[0], data.columns[1], args.export_dir, show=False,
                                format=args.plot_format, show_title=args.show_title)

        list_features = data_raw.columns.values
        if len(args.emo_dims) == 1:
            for feature in list_features:
                plots.plot_clusters(data_raw, label_df, 'labels', feature, args.export_dir, format=args.plot_format,
                                    show_title=args.show_title)
        elif len(args.emo_dims) == 2:
            for i in range(0, len(list_features), 2):
                plots.plot_clusters(data_raw, label_df, list_features[i], list_features[i + 1], args.export_dir,
                                    format=args.plot_format, show_title=args.show_title)

    # Settings and Export
    if args.export_as_labels and args.export_dir is not None:
        cluster_eval.export_results_as_labels(args.export, nan_indices, args.output_path, args.label_reference_path)

    settings = pd.DataFrame(columns=["Features", "TSNE", "PCA", "Algorithm", "K", "Linkage", "EPS", "min_cluster_size",
                                     "min_samples", "distance_thr", "remove_noise", "SOM", "Neurons", "topology"])

    settings = settings.append(
        {"Features": args.features, "TSNE": args.tsne, "PCA": args.pca, "Algorithm": args.algorithm,
         "K": args.k, "Linkage": args.linkage, "EPS": args.eps, "min_samples": args.min_samples,
         "distance_thr": args.distance_thr, "Remove_Noise": args.remove_noise, "SOM": args.som,
         "Neurons": args.neurons, "topology": args.topology}, ignore_index=True)
    ut.export_to_csv(args.export_dir, args.export, settings, results, cluster_names)
    if args.append is not None:
        ut.append_to_csv(args.append, settings, results)


def setup_export_folder_and_logger(args):
    if args.export is not None:
        args.export_dir = os.path.join(args.output_path, args.export)
        if not os.path.exists(args.export_dir):
            os.makedirs(args.export_dir)
        sys.stdout = ut.Logger(os.path.join(args.export_dir, 'log.txt'))
    else:
        args.export_dir = None


def generate_name_from_args(args):
    name = '+'.join(args.features)
    if args.standardised:
        name += '-st'

    if args.pca:
        name += f"_pca-{args.pca}"
    elif args.tsne:
        name += f"_tsne-{args.pca}"
    elif args.som:
        name += "_som"

    name += f"_{args.algorithm}"

    name += f"_{'+'.join(args.partitions)}-clustered"

    if not args.dbscan:
        name += f"_{args.k}-class"

    if len(args.emo_dims) == 1:
        name += f"_{args.emo_dims[0]}"

    return name
