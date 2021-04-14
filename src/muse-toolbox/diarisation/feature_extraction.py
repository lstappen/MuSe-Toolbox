import glob
import os
from pathlib import Path
from time import time

import natsort
import numpy as np
import pandas as pd

import config as cfg


def prepare_data(emotion_name, partition_path, segment_info_path, annotation_path):
    # read all files with the annotations per video
    videos = []
    partition_df = pd.read_csv(partition_path)
    partitions = {vid: partition for vid, partition in zip(partition_df['Id'], partition_df['Proposal'])}

    seg_info_filenames = glob.glob(os.path.join(segment_info_path, "*.csv"))
    seg_info_filenames = natsort.natsorted(seg_info_filenames)
    video_ids = [int(Path(x).stem) for x in seg_info_filenames]
    filenames_path = os.path.join(annotation_path, emotion_name)
    filenames = [os.path.join(filenames_path, f"{vid}.csv") for vid in video_ids]

    for video_id, filename, seg_info_filename in zip(video_ids, filenames, seg_info_filenames):
        video_df = pd.read_csv(filename, index_col=None)
        seg_info_df = pd.read_csv(seg_info_filename, index_col=None)
        segments = []
        for start, end, seg_id in zip(seg_info_df['start'], seg_info_df['end'], seg_info_df['segment_id']):
            start /= 1000.
            end /= 1000.
            # print(f"Start: {start}, end: {end}")
            segment = video_df.loc[(video_df['timestamp'] >= start) & (video_df['timestamp'] < end)].copy()
            segments.append((int(seg_id), partitions[video_id], segment))
        videos.append((video_id, segments))

    return videos


def calculate_statistics(videos, emotion_name, output_path):

    columns = [cfg.VIDEO_ID_KEY, cfg.SEGMENT_ID_KEY, cfg.PARTITION_KEY] + list(cfg.FEATURE_FUNCS.keys())
    dataset = {feature: [] for feature in columns}

    for video_id, segments in videos:
        print(f"Vid {video_id} ({len(segments)} segments)")
        for seg_id, partition, segment in segments:
            #print(f"Vid {video_id}, Seg {seg_id} (len {segment.values[:, 1].shape[0]})")

            dataset[cfg.VIDEO_ID_KEY].append(video_id)
            dataset[cfg.SEGMENT_ID_KEY].append(seg_id)
            dataset[cfg.PARTITION_KEY].append(partition)

            for feature in cfg.FEATURE_FUNCS.keys():
                feature_data = cfg.FEATURE_FUNCS[feature](segment.values[:, 1])
                feature_data = np.around(feature_data, decimals=6)
                dataset[feature].append(feature_data)

    data = pd.DataFrame(dataset)
    data.columns = [f"{col}_{emotion_name}" if col in cfg.FEATURE_FUNCS.keys() else col for col in data.columns]
    data.to_csv(os.path.join(output_path, f'{emotion_name}_segment_features.csv'), index=False)
    return data


if __name__ == "__main__":
    t_start = time()

    emo_dims = ['arousal', 'valence']

    for emo_dim in emo_dims:
        vids = prepare_data(emo_dim, cfg.PARTITION_PATH, cfg.SEGMENTS_TOPIC_REFERENCE_PATH, cfg.ANNOTATION_PATH)
        calculate_statistics(vids, emo_dim, cfg.DATA_PATH)

    print(f"Time taken: {round(time() - t_start, 1)} sec")
