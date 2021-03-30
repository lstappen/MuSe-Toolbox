import pandas as pd
import numpy as np
import glob

from collections import defaultdict
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from gold_standard.smoothing import smooth_data


def get_scalers(data_dir, ts, args, mapping):
    # Returns the StandardScaler and the MinMaxScaler fitted to all of the data per annotator

    data_for_std = defaultdict(list)  # collect all annotations per unique annotator
    for path in glob.glob(data_dir):
        path = path.replace("\\", '/')
        df = pd.read_csv(path)
        # smooth
        smoothed_sample, annotators = smooth_data(df, args.pre_smoothing, args.pre_smoothing_window, ts)

        for a, s in zip(annotators, smoothed_sample):
            data_for_std[mapping[a]].append(s)

    for a, a_data in data_for_std.items():
        data_for_std[a] = np.concatenate(a_data, axis=0).reshape(-1, 1)

    # One scaler for each annotator
    std_scalers = {int(a): StandardScaler() for a in data_for_std.keys()}
    minmax_scalers = {int(a): MinMaxScaler(feature_range=(-1, 1)) for a in data_for_std.keys()}

    for a, scaler in std_scalers.items():  # Fit scalers (first StandardScaler, then MinMaxScaler)
        scaler.fit(data_for_std[a])
        scaled_data = scaler.transform(data_for_std[a])
        minmax_scalers[a].fit(scaled_data)

    return std_scalers, minmax_scalers


def preprocess_signal(pre_smoothing, pre_smoothing_window, std_annos_per_sample, std_annos_all_samples, df, ts,
                      mapping=None, std_scalers=None, minmax_scalers=None, aligned=False):
    # Apply data preprocessing to the annotation signals

    smoothed_sample, annotators = smooth_data(df, pre_smoothing, pre_smoothing_window, ts, aligned=aligned)  # smoothing

    if std_annos_per_sample:  # standardize annotators per data sample (e.g., video)
        smoothed_sample = preprocessing.scale(smoothed_sample, axis=1)

    elif std_annos_all_samples:  # standardize annotators over all data samples
        assert std_scalers is not None and minmax_scalers is not None and mapping is not None
        samples = []
        for a, s in zip(annotators, smoothed_sample):
            if a in mapping.keys():  # check if annotator should be used
                scaled_s = std_scalers[mapping[a]].transform(s.reshape(-1, 1))
                scaled_s = minmax_scalers[mapping[a]].transform(scaled_s)
                samples.append(scaled_s)
        smoothed_sample = [s.reshape(-1) for s in samples]  # list of numpy arrays
    return smoothed_sample, annotators
