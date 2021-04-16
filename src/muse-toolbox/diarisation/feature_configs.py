import numpy as np
import tsfresh.feature_extraction.feature_calculators as tsf

VIDEO_ID_KEY = 'video_id'
SEGMENT_ID_KEY = 'segment_id'
PARTITION_KEY = 'partition'

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
    'mean_abs_change': lambda x: tsf.mean_abs_change(x),
    'mean_change': lambda x: tsf.mean_change(x),
    'mean_sec_derivative_central': lambda x: tsf.mean_second_derivative_central(x),
    'skewness': lambda x: tsf.skewness(x),
    'kurtosis': lambda x: tsf.kurtosis(x),
    'first_location_of_maximum': lambda x: tsf.first_location_of_maximum(x),
    'first_location_of_minimum': lambda x: tsf.first_location_of_minimum(x),
    'last_location_of_maximum': lambda x: tsf.last_location_of_maximum(x),
    'last_location_of_minimum': lambda x: tsf.last_location_of_minimum(x),
    'percentage_of_reoccurring_datapoints_to_all_datapoints': lambda x:
    tsf.percentage_of_reoccurring_datapoints_to_all_datapoints(x),
    'rel_energy': lambda x: tsf.abs_energy(x) / tsf.length(x),
    'rel_sum_of_changes': lambda x: tsf.absolute_sum_of_changes(x) / tsf.length(x),
    'rel_number_crossing_0': lambda x: tsf.number_crossing_m(x, 0) / tsf.length(x),
    'rel_number_peaks': lambda x: tsf.number_peaks(x, 10) / tsf.length(x),
    'rel_long_strike_below_mean': lambda x: tsf.longest_strike_below_mean(x) / tsf.length(x),
    'rel_long_strike_above_mean': lambda x: tsf.longest_strike_above_mean(x) / tsf.length(x),
    'rel_count_below_mean': lambda x: tsf.count_below_mean(x) / tsf.length(x)
}

# Selected feature subsets for unsupervised clustering
FEATURE_SETS = {
    'set0': ['mean', 'median', 'std', 'percentile_5', 'percentile_10', 'percentile_25', 'percentile_33',
             'percentile_66', 'percentile_75', 'percentile_90', 'percentile_95', 'rel_energy', 'rel_sum_of_changes',
             'rel_number_crossing_0', 'rel_number_peaks', 'rel_long_strike_below_mean',
             'rel_long_strike_above_mean', 'rel_count_below_mean',
             'percentage_of_reoccurring_datapoints_to_all_datapoints'],
    'set1': ['std', 'percentile_5', 'percentile_10', 'percentile_75', 'percentile_90', 'percentile_95',
             'rel_energy', 'rel_sum_of_changes', 'rel_number_crossing_0', 'rel_number_peaks',
             'rel_long_strike_below_mean', 'rel_long_strike_above_mean', 'rel_count_below_mean', 'length'],
    'set2': ['std', 'rel_energy', 'rel_sum_of_changes', 'rel_number_peaks', 'rel_long_strike_below_mean',
             'rel_long_strike_above_mean', 'rel_count_below_mean'],
    'set3': ['std', 'percentile_10', 'median', 'percentile_90', 'rel_energy', 'rel_sum_of_changes',
             'rel_number_crossing_0', 'rel_number_peaks', 'rel_long_strike_below_mean',
             'rel_long_strike_above_mean', 'rel_count_below_mean'],
    'set4': ['mean', 'median', 'std', 'percentile_5', 'percentile_10', 'percentile_25', 'percentile_33',
             'percentile_66', 'percentile_75', 'percentile_90', 'percentile_95']
}
