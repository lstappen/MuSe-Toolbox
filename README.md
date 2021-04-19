_MuSe-Toolbox_ a Python-based open-source toolkit for creating a variety of continuous and discrete emotion gold standards. 
In a single framework, we unify a wide range of fusion methods, such as Estimator Weighted Evaluator(EWE), DTW-Barycenter Averaging (DBA), and Generic-Canonical Time Warp-ing (GCTW), as well as providing an implementation of Rater Aligned Annotation Weighting (RAAW). 
The latter method, RAAW, aligns the annotations in a translation-invariant way before weighting them based on inter-rater agreement between the raw annotations. \
In addition, the _MuSe-Toolbox_ provides the functionality to run exhaustive searches for meaningful class clusters in the continuous gold standards.
For this, signal characteristics are extracted, which are then clustered to create cluster classes in an unsupervised fashion.
For a better understanding of the proposed clusters, the toolbox also offers _expressive profiling_ options to make human interpretation of the cluster characteristics easier, e.g. statistical analysis, and visualisations.

Please direct any questions or requests to contact.muse2020[@]gmail.com or stappen[@]ieee.org or via PR.


# Citing
If you use MuSe-Toolbox or any code from MuSe-Toolbox in your research work, you are kindly asked to acknowledge the use 
of MuSe-Toolbox in your publications. _TODO: citations_

> citation

```
@inproceedings{musefusebox,
...
}
```

# Installation


## Dependencies

* Python 3.7
* For CTW you need to install Octave 5.2.0 and check that 
    `os.environ['OCTAVE_EXECUTABLE'] == 'C:/Octave/Octave-5.2.0/mingw64/bin/octave-cli.exe'`
    Find older Octave versions under: https://ftp.gnu.org/gnu/octave/

## Installing the python package
* We recommend the usage of a virtual environment for the MuSe-Toolbox installation.
    ```bash 
    python3 -m venv muse_virtualenv
    ```
    Activate venv using:
    - Linux
    ```bash 
     source muse_virtualenv/bin/activate
    ```
    - Windows
    ```bash 
    muse_virtualenv\Scripts\activate.bat
    ```
    Deactivate with:
    ```bash 
     deactivate
    ```
* Once the virtual environment is activated, install the dependencies in requirements.txt with (later update with install using pip) _TODO: pip installation_
    ```bash 
    pip -r requirements.txt
    ```

The Installation is now complete.

# Configuration
Go through the tutorials for usage examples of the toolkit. _TODO: tutorials/examples_

Run with:
    ```
    python main.py [gold_standard, diarisation]
    ```
## Commandline Options: Gold Standard Generation
... _TODO: add cli options_
## Commandline Options: Diarisation

### Segment-level feature extraction (pre-processing step)

Example call:
`python src/muse-toolbox feature_extraction -dim arousal -inp examples/processed -seg_info examples/metadata/segment_info -partition examples/metadata/partition.csv -out output`

| Arg               | Type           | Description  | Options | Default | Example |
| -------------     |:-------------:| -----| ----------| ---- | ---- |
| -inp --input_path      | str | Path to directory that contains `dimension` folder. The `dimension` folder contains csv files with the fused annotations from the gold_standard module.| | `data\` | |
| -out --output_path     | str | Path where the extracted segment-level features are saved (will be saved as segment_features_<dimension>.csv) | | `output\` | |
| -dims --dimensions    | list(str)      | Perform feature extraction for the specified emotion dimensions| | `[arousal, valence]` | `valence` |
| -seg_info --segment_info_path     | str | Path to directory that contains segment id information | | `segment_info\` | |
| -partition --partition_path     | str | Path to the csv file that contains partition information  | | `partition.csv` | |

### Diarisation (class creation based on segment-level features)

Example call:
`python src/muse-toolbox diarisation -inp "../../output" -out "../../output" --features set_basic --emo_dims arousal -std --export clustering_test --pca 2 --kmeans --k 3 --export_as_labels -label_ref "../../examples/metadata/segment_info" --min_class_thr 25 --plot all --plot_format pdf`

#### General parameters

| Arg               | Type           | Description  | Options | Default | Example |
| -------------     |:-------------:| -----| ----------| ---- | ---- |
| -inp --input_path      | str | Path to directory that contains segment_features_<dimension>.csv (as created by the feature_extraction sub-module).| | `data\` | |
| -out --output_path     | str | Base path where the clustering output will be saved | | `output\` | |
| --features    | list(str)      | List of features to consider for clustering. For convenience, use `all` to select all available features, or one of the pre-sets defined in feature_configs.py | | | `all`, `set_basic`, `mean median rel_energy` |
| -std --standardised    | bool      | Standardise features before clustering | True or False | False | |
| -dims --emo_dims    | list(str)      | Perform clustering for the specified emotion dimensions | | `[arousal, valence]` | `valence` |
| --partitions    | list(str)      | List of partitions to use for clustering, the rest will be predicted. Use `all` to perform clustering on all data | | `[all]` | `train` |
| --plot    | list(str)      | Plot different visualisations of the clustering result | `none`, `all`, `point_clouds`, `distinctive_features_single`, `distinctive_features_combined`, `corr`, `corr_abs` | `[all]` | `point_clouds corr_abs` |
| --plot_format    | str      | Data type to save the plots as | | `png` | `svg` |
| --plot_no_title    | bool      | If selected, does not add a title above each plot | True or False | False | |
| --export    | str      | Name of the folder in which the results, settings and plots are saved. For convenience, use `auto` to auto-generate a descriptive name based on clustering settings | | | `kmeans_test` |
| --export_as_labels    | bool      | Save results in several csv files (mapped to videos) for further use as labels | True or False | False | |
| -label_ref --label_reference_path     | str | Path to directory that contains segment id information. Will be used as reference when creating labels from results (required if `export_as_labels` is selected) | | `segment_info\` | |
| --append    | str      | Appends the results and settings to a csv file | | | |
| --cluster_seed    | int      | Seed used for any random initialisations of the clustering algorithms | | 301 | 123 |
| --min_class_thr    | float      | Minimum size of smallest class (as factor of chance level) for the clustering output to be considered valid | | None | 0.5 |
| --reduce_separately    | bool      | Perform the feature reduction step separately for each emotion dimension given | True or False | False | |


#### Feature reduction (mutually exclusive)

| Arg               | Type           | Description  | Options | Default | Example |
| -------------     |:-------------:| -----| ----------| ---- | ---- |
| --pca    | int      | Reduce the features with PCA in advance to a certain number | | | 3 |
| --tsne    | int      | Reduce the features with t-SNE in advance to a certain number | | | 2 |
| --som    | bool      | Use a self-organising map (SOM) for feature selection | True or False | False | |

#### Clustering algorithm (mutually exclusive)

| Arg               | Type           | Description  | Options | Default | Example |
| -------------     |:-------------:| -----| ----------| ---- | ---- |
| --kmeans    | bool      | Apply the k-means algorithm | True or False | False | |
| --dbscan    | bool      | Apply the DBSCAN algorithm | True or False | False | |
| --aggl    | bool      | Apply the Agglomerative Clustering algorithm | True or False | False | |
| --gmm    | bool      | Apply Gaussian Mixture Model clustering | True or False | False | |
| --fuzzyCMeans    | bool      | Apply the fuzzy c-means algorithm | True or False | False | |
| --plotHistBox    | bool      | Plot the histogram and the boxplot for the segments (instead of clustering) | True or False | False | |

#### Specific settings for Agglomerative Clustering

| Arg               | Type           | Description  | Options | Default | Example |
| -------------     |:-------------:| -----| ----------| ---- | ---- |
| --linkage      | str | Define the used linkage criterion.| `ward`, `complete`, `average`, `single` | `ward` | |
| --distance_thr    | float      | The linkage distance threshold above which, clusters will not be merged. | | None | |

#### Specific settings for DBSCAN

| Arg               | Type           | Description  | Options | Default | Example |
| -------------     |:-------------:| -----| ----------| ---- | ---- |
| --eps    | float      | Defines the eps value for DBSCAN. | | 0.2 | |
| --min_samples    | int      | The number of samples in a neighborhood for a point to be considered as a core point. | | 5 | |
| --remove_noise    | bool      | Remove the noise from any plots | True or False | False | |

#### Settings for other clustering algorithms

| Arg               | Type           | Description  | Options | Default | Example |
| -------------     |:-------------:| -----| ----------| ---- | ---- |
| --k    | int      | Number of clusters (used by all algorithms except DBSCAN). | | 7 | |
| --m    | int      | Define the fuzzifier (used by fuzzy c-means). | | 2 | |

#### Settings for SOM feature selection

| Arg               | Type           | Description  | Options | Default | Example |
| -------------     |:-------------:| -----| ----------| ---- | ---- |
| --som_seed    | int      | Seed used for random initialisation of SOM weights. | | 301 | |
| --weights    | str      | Initialisation method of the weights. | `random`, `pca` | `random` | |
| --topology    | str      | Topology of the SOM. | `rectangular`, `hexagonal` | `hexagonal` | |
| --neighborhood_function    | str      | The used neighborhood function. | `gaussian`, `mexican_hat`, `bubble`, `triangle` | `gaussian` | |
| --dist    | str      | The distance function. | `euclidean`, `cosine`, `manhattan` | `euclidean` | |
| --sigma    | float      | Spread of the neighborhood function. | | 1.0 | |
| --alpha    | float      | The learning rate. | | 0.9 | |
| --iter    | int      | The number of training iterations. | | 1000 | |
| --neurons    | int      | The number of output neurons. If none is provided, the optimal number of neurons will be calculated. | | | 100 |



(c) Chair of Embedded Intelligence for Health Care and Wellbeing, University of Augsburg. Published under GNU General Public license.
