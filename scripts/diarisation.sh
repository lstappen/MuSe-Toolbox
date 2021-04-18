#!/bin/sh

# Creating segment-level classes with unsupervised clustering based on gold-standard annotations

input=examples/processed  # path to input gold standard annotations
segment_info=examples/metadata/segment_info
partition_info=examples/metadata/partition.csv
output=output/diarisation

emotion_dim=arousal  # emotion dimension to use for clustering
plot_format=png  # file format to save the plots as

######################

# Preparation step: extract segment-level features

python src/muse-toolbox feature_extraction -inp $input -out $output -dim $emotion_dim -seg_info $segment_info -partition $partition_info

######################

# 1. all features, all data, no pca, kmeans
export=example_1  # this will be the name of the folder where all results will be saved in
features=all  # use all features available in the input csv file
clustering_algo=kmeans  # use k-means clustering algorithm
n_clusters=5  # number of output classes
partitions=all  # perform clustering on all data
python src/muse-toolbox diarisation -inp $output -out $output --features $features -dims $emotion_dim --partitions $partitions -std --$clustering_algo --k $n_clusters --export $export --export_as_labels -label_ref $segment_info --plot all --plot_format $plot_format

# 2. set_ext, train-only, pca 2, gmm
export=example_2
features=set_ext  # use a feature preset as defined in src/muse-toolbox/diarisation/feature_configs.py
clustering_algo=gmm  # use gaussian mixture model for clustering
n_clusters=5
partitions=train  # perform clustering on the training partition only (the other samples will be assigned to the closest cluster afterwards)
pca=2  # apply Principal Component Analysis (PCA) first, and then use the 2 principal components for clustering
python src/muse-toolbox diarisation -inp $output -out $output --features $features -dims $emotion_dim --partitions $partitions -std --pca $pca --$clustering_algo --k $n_clusters --export $export --export_as_labels -label_ref $segment_info --plot all --plot_format $plot_format


# 3. mean+median, train-only, no pca, fuzzyCMeans
export=example_3
features=mean median  # use only a handpicked selection of features for clustering
clustering_algo=fuzzyCMeans  # use fuzzy c-means clustering algorithm
n_clusters=5
partitions=train
python src/muse-toolbox diarisation -inp $output -out $output --features $features -dims $emotion_dim --partitions $partitions -std --$clustering_algo --k $n_clusters --export $export --export_as_labels -label_ref $segment_info --plot all --plot_format $plot_format


