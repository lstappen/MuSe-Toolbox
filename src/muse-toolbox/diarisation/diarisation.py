import pandas as pd
import os
import sys
from tabulate import tabulate
from itertools import chain

import diarisation.evaluation as evaluation
from diarisation.feature_reduction import apply_feature_reduction, univariate_feature_selection
import diarisation.cluster_naming as cluster_naming
import diarisation.plots as plots
import diarisation.utils as ut
import diarisation.config as config


def diarisation(args):
    if args.export is not None and args.export == 'auto':
        args.export = generate_name_from_params(args)

    print(args.export)
    setup_export_folder_and_logger(args)

    # Feature selection and reduction:
    segments = ut.load_data(args.emo_dims, args.seg_type, args.standardised)
    available_features = list(chain.from_iterable([list(segments_emo.columns) for segments_emo in segments]))
    features = ut.extract_features_cli(args.features, available_features, args.emo_dims)
    data, data_mean, nan_indices, partitions = ut.select_features(features, segments, args.emo_dims)
    data_raw = data.copy()

    if f'length_{args.emo_dims[0]}' in data.columns:
        data.drop(columns=[f'length_{args.emo_dims[0]}'])

    print(f"Data shape: {data.shape}")

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
        print(f"Data columns (after feature reduction): {data.columns}")
    else:
        data = apply_feature_reduction(data, args)
        if args.tsne is not None or args.pca is not None:
            n_components = args.tsne if args.tsne is not None else args.pca
            data.columns = [f'c_{i}' for i in range(n_components)]
    if args.som:
        data_raw = data.copy()

    # Plot Hist and Box
    if args.plotHistBox and len(args.emo_dims) == 2:
        plots.plot_histogram(data_mean[f"mean_{args.emo_dims[1]}"], data_mean[f"mean_{args.emo_dims[0]}"])
        plots.plot_boxplot(data_mean[f"mean_{args.emo_dims[1]}"], data_mean[f"mean_{args.emo_dims[0]}"])

    # Clustering
    labels, fpc = ut.apply_clustering(data, args, partitions)

    # Cluster Evaluation
    cluster_eval = evaluation.Evaluation(data=data_raw, cluster_labels=labels, fpc=fpc, plot=args.plot,
                                         remove_noise=args.remove_noise, export_dir=args.export_dir)
    results = cluster_eval.get_results()

    print(tabulate(results, headers=list(results.columns.values), tablefmt="grid"))

    # Cluster Naming
    naming = cluster_naming.ClusterNaming(data_mean, labels, args.emo_dims, args.plot, args.export_dir)
    cluster_names = naming.naming()
    # naming.plot_attributes()
    print(tabulate(cluster_names, headers=list(cluster_names.columns.values), tablefmt="grid"))

    # apply rule of thumb threshold to only accept reasonable class distributions
    if args.min_class_thr is not None:
        smallest_class = cluster_names['data_per_cluster'].values.min()
        # smallest class should have at least args.min_class_thr of chance level
        rule_of_thumb_threshold = args.min_class_thr / args.k
        distr_is_ok = smallest_class >= rule_of_thumb_threshold
        print(args.export)
        print(f"Smallest class ({'not ' if not distr_is_ok else ''}OK): {smallest_class}")
        if not distr_is_ok:
            print(f"Time to abort...")
            return

    if args.export_as_labels and args.export_dir is not None:
        cluster_eval.export_results_as_labels(args.export, nan_indices, args.seg_type)

    # more cluster analysis TODO: make these optional via args
    labels = pd.DataFrame(labels, columns=["labels"])
    plots.plot_clusters(data, labels, data.columns[0], data.columns[1], args.export_dir, True)
    plots.plot_target_correlation(data_raw, labels, args.export_dir, True)
    #plots.plot_target_correlation(data_raw, data[data.columns[0]], params.export_dir)
    #plots.plot_most_distinctive_features_per_cluster(data_raw, labels, show=True, export_dir=params.export_dir,
    #                                                 standardize=True)
    plots.plot_most_distinctive_features_over_all_clusters(data_raw, labels, args.emo_dims[0], show=True, export_dir=args.export_dir,
                                                           standardize=False)
    plots.plot_most_distinctive_features_over_all_clusters(data_raw, labels, args.emo_dims[0], show=True,
                                                           export_dir=args.export_dir,
                                                           standardize=True)

    list_features = data_raw.columns.values
    if len(args.emo_dims) == 1:
        for feature in list_features:
            plots.plot_clusters(data_raw, labels, 'labels', feature, args.export_dir)
    elif len(args.emo_dims) == 2:
        # univariate_feature_selection(data_raw, labels, list_features)
        half_len = int(len(list_features) / 2)
        for i in range(half_len):
            plots.plot_clusters(data_raw, labels, list_features[i], list_features[i + half_len], args.export_dir)

    # Settings and Export
    settings = pd.DataFrame(columns=["Features", "TSNE", "PCA", "Algorithm", "K", "Linkage", "EPS", "min_cluster_size",
                                     "min_samples", "distance_thr", "remove_noise", "SOM", "Neurons", "topology"])
    if args.kmeans:
        algorithm_name = "Kmeans"
    elif args.dbscan:
        algorithm_name = "DBSCAN"
    elif args.aggl:
        algorithm_name = "Agglomerative"
    elif args.gmm:
        algorithm_name = "GMM"
    elif args.fuzzyCMeans:
        algorithm_name = "Fuzzy-C-Means"

    settings = settings.append(
        {"Features": args.features, "TSNE": args.tsne, "PCA": args.pca, "Algorithm": algorithm_name,
         "K": args.k, "Linkage": args.linkage, "EPS": args.eps, "min_samples": args.min_samples,
         "distance_thr": args.distance_thr, "Remove_Noise": args.remove_noise, "SOM": args.som,
         "Neurons": args.neurons, "topology": args.topology}, ignore_index=True)
    ut.export_to_csv(args.export_dir, args.export, settings, results, cluster_names)
    if args.append is not None:
        ut.append_to_csv(args.append, settings, results)


def setup_export_folder_and_logger(args):
    if args.export is not None:
        args.export_dir = os.path.join(config.OUTPUT_FOLDER, args.export)
        if not os.path.exists(args.export_dir):
            os.makedirs(args.export_dir)
        sys.stdout = ut.Logger(os.path.join(args.export_dir, 'log.txt'))
        print(f"Args: {args}")
    else:
        args.export_dir = None


def generate_name_from_params(params):
    name = '+'.join(params.features)
    if params.seg_type == 'wild':
        name += '-wild'
    if params.standardised:
        name += '-st'

    if params.pca:
        name += f"_pca-{params.pca}"
    elif params.tsne:
        name += f"_tsne-{params.pca}"
    elif params.som:
        name += "_som"

    if params.kmeans:
        name += "_kmeans"
    elif params.aggl:
        name += "_aggl"
    elif params.gmm:
        name += "_gmm"
    elif params.fuzzyCMeans:
        name += "_fuzzyCMeans"
    elif params.dbscan:
        name += "_dbscan"

    if params.partitions[0] == 'all':
        name += '-all'

    name += f"_{params.k if params.k is not None else 'x'}-class"

    if len(params.emo_dims) == 1:
        name += f"_{params.emo_dims[0]}"

    return name
