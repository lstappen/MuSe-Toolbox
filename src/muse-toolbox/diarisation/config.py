import socket
import os
import numpy as np
import tsfresh.feature_extraction.feature_calculators as tsf

# path
if 'eihw-gpu' in socket.gethostname():
    BASE_PATH = '/nas/student/BenjaminSertolli/PM/data'
    LABEL_EXPORT_PATH = '/nas/student/EmCaR/8_MuSe2021/confidential/c2_muse_sent/label_segments'
else:
    BASE_PATH = r'E:\Projektmodul_data'
    LABEL_EXPORT_PATH = os.path.join(BASE_PATH, 'c2_muse_sent', 'label_segments')

OUTPUT_FOLDER = './output'

# paths for unsupervised clustering
DATA_PATH = os.path.join(BASE_PATH, 'diari_data')
LABEL_BASE_PATH = os.path.join(DATA_PATH, 'label_base_')

# paths for feature extraction
ANNOTATION_WILD_PATH = os.path.join(BASE_PATH, 'c2_muse_sent', 'raw', 'annotations', 'task1', 'label_segments')
ANNOTATION_GS_PATH = os.path.join(BASE_PATH, 'c2_muse_sent', 'raw', 'annotations', 'gold_standard')
SEGMENTS_TOPIC_REFERENCE_PATH = os.path.join(DATA_PATH, 'label_base_topic')
METADATA_PATH = os.path.join(DATA_PATH, 'metadata.csv')
PARTITION_PATH = os.path.join(BASE_PATH, 'metadata', 'partition.csv')

# Selected feature subsets for unsupervised clustering
FEATURE_SETS = {
    'new_only': ['abs_energy', 'abs_sum_of_changes', 'mean_abs_change', 'mean_change', 'mean_sec_derivative_central',
                 'number_crossing_m', 'number_peaks', 'kurtosis', 'long_strike_below_mean',
                 'long_strike_above_mean', 'count_below_mean', 'first_location_of_maximum', 'first_location_of_minimum',
                 'last_location_of_maximum', 'last_location_of_minimum',
                 'percentage_of_reoccurring_datapoints_to_all_datapoints'],
    'set0_abs': ['mean', 'median', 'std', 'percentile_5', 'percentile_10', 'percentile_25', 'percentile_33',
             'percentile_66', 'percentile_75', 'percentile_90', 'percentile_95', 'abs_energy', 'abs_sum_of_changes',
             'number_crossing_m', 'number_peaks', 'long_strike_below_mean', 'long_strike_above_mean',
             'count_below_mean', 'percentage_of_reoccurring_datapoints_to_all_datapoints', 'length'],
    'set1_abs': ['std', 'percentile_5', 'percentile_10', 'percentile_75', 'percentile_90', 'percentile_95', 'abs_energy',
             'abs_sum_of_changes', 'number_crossing_m', 'number_peaks', 'long_strike_below_mean',
             'long_strike_above_mean', 'count_below_mean'],
    'set2_abs': ['std', 'abs_energy', 'abs_sum_of_changes', 'number_peaks', 'long_strike_below_mean',
             'long_strike_above_mean', 'count_below_mean'],
    'set0_rel': ['mean', 'median', 'std', 'percentile_5', 'percentile_10', 'percentile_25', 'percentile_33',
                 'percentile_66', 'percentile_75', 'percentile_90', 'percentile_95', 'rel_energy', 'rel_sum_of_changes',
                 'rel_number_crossing_m', 'rel_number_peaks', 'rel_long_strike_below_mean',
                 'rel_long_strike_above_mean', 'rel_count_below_mean',
                 'percentage_of_reoccurring_datapoints_to_all_datapoints', 'length'],
    'set1_rel': ['std', 'percentile_5', 'percentile_10', 'percentile_75', 'percentile_90', 'percentile_95',
                 'rel_energy', 'rel_sum_of_changes', 'rel_number_crossing_m', 'rel_number_peaks',
                 'rel_long_strike_below_mean', 'rel_long_strike_above_mean', 'rel_count_below_mean', 'length'],
    'set2_rel': ['std', 'rel_energy', 'rel_sum_of_changes', 'rel_number_peaks', 'rel_long_strike_below_mean',
                 'rel_long_strike_above_mean', 'rel_count_below_mean', 'length'],
    'set3_rel': ['std', 'percentile_10', 'median', 'percentile_90', 'rel_energy', 'rel_sum_of_changes',
                 'rel_number_crossing_m', 'rel_number_peaks', 'rel_long_strike_below_mean',
                 'rel_long_strike_above_mean', 'rel_count_below_mean', 'length'],
}

# Feature extraction: available features and functions to apply
FEATURE_FUNCS = {
    'mean': lambda x: np.mean(x),
    'median': lambda x: tsf.median(x),
    'std': lambda x: np.std(x),
    'percentile_5': lambda x: np.quantile(x, 0.05),
    'percentile_10': lambda x: np.quantile(x, 0.1),
    'percentile_25': lambda x: np.quantile(x, 0.25),
    'percentile_33': lambda x: np.quantile(x, 0.33),
    'percentile_66': lambda x: np.quantile(x, 0.66),
    'percentile_75': lambda x: np.quantile(x, 0.75),
    'percentile_90': lambda x: np.quantile(x, 0.9),
    'percentile_95': lambda x: np.quantile(x, 0.95),
    # 'mean_per_video': lambda x: statistics.mean(x),
    'abs_energy': lambda x: tsf.abs_energy(x),
    'abs_sum_of_changes': lambda x: tsf.absolute_sum_of_changes(x),
    'mean_abs_change': lambda x: tsf.mean_abs_change(x),
    'mean_change': lambda x: tsf.mean_change(x),
    'mean_sec_derivative_central': lambda x: tsf.mean_second_derivative_central(x),
    'number_crossing_0': lambda x: tsf.number_crossing_m(x, 0),
    'number_peaks': lambda x: tsf.number_peaks(x, 10),
    'skewness': lambda x: tsf.skewness(x),
    'kurtosis': lambda x: tsf.kurtosis(x),
    'long_strike_below_mean': lambda x: tsf.longest_strike_below_mean(x),
    'long_strike_above_mean': lambda x: tsf.longest_strike_above_mean(x),
    'count_below_mean': lambda x: tsf.count_below_mean(x),
    'first_location_of_maximum': lambda x: tsf.first_location_of_maximum(x),
    'first_location_of_minimum': lambda x: tsf.first_location_of_minimum(x),
    'last_location_of_maximum': lambda x: tsf.last_location_of_maximum(x),
    'last_location_of_minimum': lambda x: tsf.last_location_of_minimum(x),
    'percentage_of_reoccurring_datapoints_to_all_datapoints': lambda x:
    tsf.percentage_of_reoccurring_datapoints_to_all_datapoints(x),
    'length': lambda x: tsf.length(x),
    'rel_energy': lambda x: tsf.abs_energy(x) / tsf.length(x),
    'rel_sum_of_changes': lambda x: tsf.absolute_sum_of_changes(x) / tsf.length(x),
    'rel_number_crossing_0': lambda x: tsf.number_crossing_m(x, 0) / tsf.length(x),
    'rel_number_peaks': lambda x: tsf.number_peaks(x, 10) / tsf.length(x),
    'rel_long_strike_below_mean': lambda x: tsf.longest_strike_below_mean(x) / tsf.length(x),
    'rel_long_strike_above_mean': lambda x: tsf.longest_strike_above_mean(x) / tsf.length(x),
    'rel_count_below_mean': lambda x: tsf.count_below_mean(x) / tsf.length(x),
    # 'sample_entropy': lambda x: tsf.sample_entropy(x), # occasionally produces errors due to zero-division
}
VIDEO_ID_KEY = 'video_id'
SEGMENT_ID_KEY = 'segment_id'
PARTITION_KEY = 'partition'
METADATA_KEYS = ['id', 'view_count_pd', 'like_count_pd', 'dislike_count_pd', 'comment_count_pd', 'duration_sec']