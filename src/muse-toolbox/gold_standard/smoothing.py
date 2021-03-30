import numpy as np

from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d


def smooth_data(df, smooth_type, filter_frame, ts, aligned=False):
    # Smooth the annotation signals, ignore timestamp column
    data = []
    annotators = []
    for col in list(df.columns):
        if col == ts:
            continue
        # if not aligned:  # uncomment if needed
        #     data.append(df[col].dropna().values / 1000)
        # else:
        data.append(df[col].dropna().values)

        annotators.append(col)

    assert filter_frame % 2 == 1 and filter_frame >= 3
    assert smooth_type in ['savgol', 'avg', 'none']

    smoothed_data = []
    if smooth_type == 'none':
        smoothed_data = data
    else:
        for seq in data:  # smooth each annotation signal with specified smoothing method
            if smooth_type == 'savgol':
                smoothed_seq = savgol_filter(seq, filter_frame, 3)
            elif smooth_type == 'avg':
                smoothed_seq = uniform_filter1d(seq, size=filter_frame)
            smoothed_data.append(smoothed_seq)
    return smoothed_data, annotators


def smooth_fusion(y, box_pts):
    # Smooth y using a sliding covolution window of size box_pts
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return np.array(y_smooth).flatten().reshape(-1, 1)
