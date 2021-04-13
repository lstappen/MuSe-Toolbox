import datetime
import glob
import os
from pathlib import Path
from time import time

import natsort
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config as cfg


def prepare_metadata():
    all_metadata = pd.read_csv(cfg.METADATA_PATH)
    all_metadata = all_metadata[:-1]
    # create a dataframe with the important variables for the correlation analysis
    metadata = all_metadata.drop(
        ['Youtube_Id', 'ChannelTitle', 'ChannelID', 'Time Of Upload', 'url', 'Transcriptions', 'Download Completed',
         'Download Format', 'File Size', 'Size'], axis=1)
    metadata = metadata.drop(metadata.columns[0], axis=1)

    # calculate metadata per day (i.e. likes/time period)
    metadata['Date of Download'] = datetime.date(2019, 5, 25)
    metadata['Date of Upload'] = pd.to_datetime(metadata['Date of Upload'])
    metadata['Date of Download'] = pd.to_datetime(metadata['Date of Download'])
    metadata['Time period'] = metadata['Date of Download'] - metadata['Date of Upload']
    metadata['Time period'] = metadata['Time period'].dt.days

    metadata = metadata.drop(['Date of Download', 'Date of Upload'], axis=1)
    metadata['View Count pd'] = metadata['View Count'] / metadata['Time period']
    metadata['Like Count pd'] = metadata['Like Count'] / metadata['Time period']
    metadata['Dislike Count pd'] = metadata['Dislike Count'] / metadata['Time period']
    metadata['Comment Count pd'] = metadata['Comment count'] / metadata['Time period']
    metadata['Duration sec'] = metadata.apply(
        lambda x: int(x['Duration'].split(':')[0]) * 60 +
                  0 if x['Duration'].split(':')[1] == '' else int(x['Duration'].split(':')[1]), axis=1)
    metadataPd = metadata.drop(
        ['Title', 'View Count', 'Like Count', 'Dislike Count', 'Duration', 'Comment count', 'Time period'], axis=1)
    # missing values: remove the datasets with missing values
    missing_values = [12, 32, 33, 41, 44, 45, 46, 102, 182, 260, 265, 299]
    metadataReduced = metadataPd[~(metadataPd["Id"].isin(missing_values))]
    metadataReduced.columns = ['id', 'view_count_pd', 'like_count_pd', 'dislike_count_pd', 'comment_count_pd',
                               'duration_sec']
    return metadataReduced


def __prepare_data(emotion_name, segment_type):
    # read all files with the annotations per video
    videos = []
    partition_df = pd.read_csv(cfg.PARTITION_PATH)
    partitions = {vid: partition for vid, partition in zip(partition_df['Id'], partition_df['Proposal'])}

    if 'topic' in segment_type:
        seg_info_filenames = glob.glob(os.path.join(cfg.SEGMENTS_TOPIC_REFERENCE_PATH, "*.csv"))
        seg_info_filenames = natsort.natsorted(seg_info_filenames)
        video_ids = [int(Path(x).stem) for x in seg_info_filenames]
        filenames_path = os.path.join(cfg.ANNOTATION_PATH[segment_type], emotion_name)
        name_suffix = f'_{emotion_name.upper()[0]}_GS.txt' if 'ewe' in segment_type else '.csv'
        filenames = [os.path.join(filenames_path, f"{vid}{name_suffix}") for vid in video_ids]

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

    elif 'wild' in segment_type:
        filenames = glob.glob(os.path.join(cfg.ANNOTATION_PATH[segment_type], emotion_name, "*.csv"))
        filenames = natsort.natsorted(filenames)

        for filename in filenames:
            video_id = int(os.path.basename(filename).split(".")[0])
            video_df = pd.read_csv(filename, index_col=None)
            segments = [y for x, y in video_df.groupby('segment_id', as_index=False)]
            seg_ids = sorted(list(set(video_df['segment_id'].values)))
            seg_partitions = [partitions[video_id]] * len(seg_ids)
            merged = list(zip(seg_ids, seg_partitions, segments))
            videos.append((video_id, merged))

    else:
        print(f"Invalid segment type '{segment_type}'. Must be either 'wild' or 'topic'")

    return videos


def calculate_statistics(emotion_name, segment_type, metadata):
    videos = __prepare_data(emotion_name, segment_type)

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

    print(f"# rows with segment id info: {len(dataset[cfg.SEGMENT_ID_KEY])}")
    dataset = pd.DataFrame(dataset)
    print(f"# rows in Dataframe (before merging metadata): {len(dataset.index)}")

    # merge the features with the metadata dataframe
    data = pd.merge(dataset, metadata, left_on=cfg.VIDEO_ID_KEY, right_on="id").drop("id", axis=1)
    print(f"# rows in Dataframe (after merging): {len(data.index)}")
    data.columns = [f"{col}_{emotion_name}" if col in cfg.FEATURE_FUNCS.keys() else col for col in data.columns]
    print(f"# rows in Dataframe (before writing to file): {len(data.index)}")
    data.to_csv(os.path.join(cfg.DATA_PATH, f'{emotion_name}_{segment_type}_features.csv'), index=False)
    return data


def normalise_statistics(data, segment_type, emotion_name):
    video_id = data[cfg.VIDEO_ID_KEY]
    segment_id = data[cfg.SEGMENT_ID_KEY]
    partitions = data[cfg.PARTITION_KEY]
    data_columns = data.columns

    # Standardize the date using the z-score
    data = data.drop(columns=[cfg.VIDEO_ID_KEY, cfg.SEGMENT_ID_KEY, cfg.PARTITION_KEY])  # after merge, videoID no longer needed
    stdsc = StandardScaler()
    data_scaled = stdsc.fit_transform(data)
    data_scaled = np.around(data_scaled, decimals=6)
    data_scaled = pd.DataFrame(data_scaled)
    data_scaled = pd.concat([video_id, segment_id, partitions, data_scaled], axis=1)
    data_scaled.columns = data_columns

    data_scaled.to_csv(os.path.join(cfg.DATA_PATH, f'{emotion_name}_{segment_type}_standardised_features.csv'),
                       index=False)
    return data_scaled


if __name__ == "__main__":
    t_start = time()

    seg_types = ['topic_ewe', 'topic_raaw', 'wild_ewe']  # choices: ['topic_ewe', 'topic_raaw', 'wild_ewe']
    emo_dims = ['arousal', 'valence']

    metadata = prepare_metadata()

    for seg_type in seg_types:
        for emo_dim in emo_dims:
            data = calculate_statistics(emo_dim, seg_type, metadata)
            # normalise_statistics(data, seg_type, emo_dim)

    print(f"Time taken: {round(time() - t_start, 1)} sec")
