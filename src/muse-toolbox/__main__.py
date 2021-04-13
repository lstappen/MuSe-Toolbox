import argparse

from diarisation.diarisation import diarisation
from gold_standard.gold_standard import gold_standard


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='module', help='Hey, this is our cool MuseFuseBox')

    gs_parser = subparsers.add_parser("gold_standard")
    diari_parser = subparsers.add_parser("diarisation")

    ### Aligning/Fusing several annotations to one Gold Standard
    gs_parser.add_argument('-inp', '--input_path', default="data", type=str,
                           help='specify the input directory')
    gs_parser.add_argument('-out', '--output_path', default="output", type=str,
                           help='specify the output directory')
    gs_parser.add_argument('-dim', '--dimension', type=str, default='arousal',
                           help='perform fusion for the specified emotion dimension (e.g. arousal, valence)')

    gs_parser.add_argument('-fuse', '--fusion', default="none", type=str, choices=['mean', 'ewe', 'dba', 'none'],
                           help='specify the fusion method (options: mean, ewe, dba, none)')
    gs_parser.add_argument('-align', '--alignment', default="none", type=str, choices=['ctw', 'none'],
                           help='specify the alignment method (options: ctw, none)')

    gs_parser.add_argument('--std_annos_per_sample', required=False, action='store_true',
                           help='specify whether to standardize per annotator per data sample')
    gs_parser.add_argument('--std_annos_all_samples', required=False, action='store_true',
                           help='specify whether to standardize per annotator over all data samples')
    gs_parser.add_argument('--pre_smoothing', default='savgol', type=str,
                           help='specify filter for smoothing the raw annotation signals (savgol, avg, or none)')
    gs_parser.add_argument('--pre_smoothing_window', default='5', type=int,
                           help='specify filter frame size for smoothing the raw annotation signals (odd number)')
    gs_parser.add_argument('--post_smoothing_window', default='10', type=int,
                           help='specify the conv window size for smoothing the fused annotation signal using')

    gs_parser.add_argument('--plot', required=False, action='store_true',
                           help='plot smoothed signals and predictions')
    gs_parser.add_argument('--start', default='0', type=int,
                           help='start with video x')
    gs_parser.add_argument('--end', default='-1', type=int,
                           help='end with video y')

    gs_parser.add_argument('--anno_mapping_file', default='none', type=str,
                           help='specify path to annotator mapping file if there are several ids for one annotator')
    gs_parser.add_argument('--annotators', nargs='+',
                           help='specify annotator ids that are present in data files')
    gs_parser.add_argument('--ts', default='timeStamp', type=str,
                           help='specify timestamp name that is to be found in the csv files.')
    gs_parser.add_argument('--ts_path', default=None, type=str,
                           help='Specify path that contain timestamps (necessary for files containing sequences of '
                                'different lengths).')

    # Feature selection and reduction:
    diari_parser.add_argument("--features", nargs="+", required=True,
                              help="Clustering Features (use 'all' to select all available features)")
    reduction = diari_parser.add_mutually_exclusive_group(required=False)
    reduction.add_argument("--tsne", type=int, help="Reduce the features with t-SNE in advance to a certain number ")
    reduction.add_argument("--pca", type=int, help="Reduce the features with PCA in advance to a certain number")
    reduction.add_argument("--som", action="store_true", help="Use a som for feature selection.")
    diari_parser.add_argument("--reduce_separately", action="store_true", help="Perform the feature reduction step "
                                                                               "separately for arousal and valence features")

    # Parameters for som:
    som_parameters = diari_parser.add_argument_group()
    som_parameters.add_argument("--som_seed", type=int, default=301,
                                help="Seed used for random initialisation of SOM weights")
    som_parameters.add_argument("--weights", choices=["random", "pca"], default="random",
                                help="Initialization method of the weights. The default is: random")
    som_parameters.add_argument("--topology", choices=["rectangular", "hexagonal"], default="hexagonal",
                                help="Topology of the som. The default is: hexagonal")
    som_parameters.add_argument("--neighborhood_function", choices=["gaussian", "mexican_hat", "bubble", "triangle"],
                                default="gaussian", help="The used neighborhood function. The default is: gaussian.")
    som_parameters.add_argument("--dist", choices=["euclidean", "cosine", "manhattan"],
                                help="The distance function. Default is euclidean")
    som_parameters.add_argument("--sigma", type=float, default=1.0, help="Spread of the neighborhood function")
    som_parameters.add_argument("--alpha", type=float, default=0.9, help="The learning rate.")
    som_parameters.add_argument("--iter", type=int, default=1000, help="The number of training iterations")
    som_parameters.add_argument("--neurons", type=int,
                                help="The number of output neurons. If none is provided, the optimal "
                                     "number of neurons will be calculated")

    # general parameters
    diari_parser.add_argument("--standardised", action="store_true", help="Use standardised features")
    diari_parser.add_argument("--metric", choices=["euclidean", "l1", "l2", "manhattan", "minkowski"],
                              help="The standard metric is euclidean. Use a different metric where possible ")
    diari_parser.add_argument("--plot", action="store_true", help="Visualization of the clustered data.")
    diari_parser.add_argument("--export", type=str,
                              help="Name of the csv file in which the results, settings and names are saved")
    diari_parser.add_argument("--export_as_labels", action="store_true",
                              help="save results in several csv files (mapped to videos) for further use as labels")
    diari_parser.add_argument("--append", type=str, help="Appends the results and settings to a csv file")
    diari_parser.add_argument("--remove_noise", action="store_true", help="Remove the noise from the plot")
    diari_parser.add_argument("--cluster_seed", type=int, default=301,
                              help="Seed used for any random initialisations of the clustering algorithms")
    diari_parser.add_argument("--seg_type", choices=["wild", "topic"], default="topic",
                              help="Type of data segments to choose. Default: topic")
    diari_parser.add_argument('--emo_dims', nargs='+', default=['arousal', 'valence'],
                              help='List of emotional dimensions to use, e.g. arousal, valence etc.')
    diari_parser.add_argument("--min_class_thr", type=float, default=None,
                              help="Minimum size of smallest class (as percentage of chance level) to be considered valid. "
                                   "Default: None")
    diari_parser.add_argument('--partitions', nargs='+', default=['train'],
                              help="List of partitions to use for clustering, the rest will be predicted. Use 'all' to"
                                   "perform clustering on the full available data set. Default: ['train']")
    diari_parser.add_argument("--anno_type", choices=["ewe", "raaw"], default="raaw",
                              help="Type of annotation fusion that was applied. Default: raaw")

    # Select the algorithm:
    algorithm = diari_parser.add_mutually_exclusive_group(required=True)
    algorithm.add_argument("--kmeans", action="store_true", help="Apply the k-means algorithm")
    algorithm.add_argument("--dbscan", action="store_true", help="Apply the dbscan algorithm")
    algorithm.add_argument("--aggl", action="store_true", help="Apply the Agglomerative Clustering algorithm")
    algorithm.add_argument("--gmm", action="store_true", help="Apply GMM")
    algorithm.add_argument("--fuzzyCMeans", action="store_true", help="Apply Fuzzy-c-means")
    algorithm.add_argument("--plotHistBox", action="store_true",
                           help="Plot the histogram and the boxplot for the segments")

    # Parameters for Agglomerative Clustering
    aggl = diari_parser.add_argument_group()
    # as agglomerative clustering requires a stopping criterion, the number of resulting clusters is used
    # no additional variable is introduced, instead k is used by k-means
    aggl.add_argument("--linkage", choices=["ward", "complete", "average", "single"],
                      help="Define the used linkage criterion. The default value is ward.")

    # Parameters for dbscan:
    dbscan = diari_parser.add_argument_group()
    dbscan.add_argument("--eps", type=float, default=0.2, help="Defines the eps value for dbscan")
    dbscan.add_argument("--min_samples", type=int, default=5,
                        help="The number of samples in a neighborhood for a point to be considered as a core point. ")

    # Parameters for k-means, fuzzy-c-means and agglomerative clustering
    kmeans = diari_parser.add_mutually_exclusive_group(required=False)
    kmeans.add_argument("--k", default=7, type=int, help="Number of clusters. Default value is 7.")
    # Parameters for fuzzy-c-means:
    kmeans.add_argument("--m", default=2, type=int, help="Define the fuzzifier")
    # Parameter for Agglomerative Clustering
    kmeans.add_argument("--distance_thr", type=float, default=None,
                        help="The linkage distance threshold above which, clusters will not be merged.")

    params = parser.parse_args()
    return params


args = parse_args()

if args.module == 'gold_standard':
    gold_standard(args)
elif args.module == 'diarisation':
    diarisation(args)

print('Done.')
