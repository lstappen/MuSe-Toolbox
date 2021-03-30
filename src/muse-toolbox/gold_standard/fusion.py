import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

from sklearn.preprocessing import MinMaxScaler

from gold_standard.DBA import compute_DBA
from gold_standard.EWE import compute_EWE
from gold_standard.interrater_agreement import compute_interrater_agreement
from gold_standard.smoothing import smooth_fusion
from gold_standard.utils import make_output_dirs_fusion, get_annotator_mapping
from gold_standard.preprocessing import preprocess_signal, get_scalers


def fuse(args):
    # make dirs
    output_path_method_dim, plot_path_method_dim = make_output_dirs_fusion(args.output_path, args.fusion,
                                                                           args.dimension)

    data_dir = os.path.join(args.input_path, args.dimension) + '/*.csv'
    ts = args.ts  # e.g., 'timeStamp'
    interrater = []

    if args.std_annos_all_samples:  # fit scalers to data
        mapping = get_annotator_mapping(args.anno_mapping_file, args.annotators)
        std_scalers, minmax_scalers = get_scalers(data_dir, ts, args, mapping)

    data_dirs = glob.glob(data_dir)
    args.end = len(data_dirs) if args.end == -1 else args.end
    for path in data_dirs[args.start:args.end]:  # iterate over all/part of the instances that are to be fused
        path = path.replace("\\", '/')
        sample_id = path.split('/')[-1]
        save_path = os.path.join(output_path_method_dim, sample_id)

        # check if fusion has already been conducted for this data instance
        if os.path.isfile(save_path):
            continue

        # read data and drop rows with missing values
        df = pd.read_csv(path)
        if df.isnull().values.any():
            print(f'Warning: {sample_id} contains NaN values. Dropping any rows, which contain NaN values.')
        df = df.dropna(axis=0)

        # preprocessing: smoothing + scaling
        if args.std_annos_all_samples:
            smoothed_sample, annotators = preprocess_signal(args.pre_smoothing, args.pre_smoothing_window,
                                                            args.std_annos_per_sample, args.std_annos_all_samples, df,
                                                            ts, mapping, std_scalers, minmax_scalers, args.aligned)
        else:
            smoothed_sample, annotators = preprocess_signal(args.pre_smoothing, args.pre_smoothing_window,
                                                            args.std_annos_per_sample, args.std_annos_all_samples, df,
                                                            ts, aligned=args.aligned)

        # EWE Fusion
        if args.fusion == 'ewe':
            fused_sample = compute_EWE([np.expand_dims(np.array(smoothed_sample), 2)])[0]

        # DBA Fusion
        if args.fusion == 'dba':
            fused_sample = compute_DBA(smoothed_sample, timestamps=df[df.columns[0]].values, n_iterations=10,
                                       plot_iterations=False)

        # Mean Fusion
        if args.fusion == 'mean':
            fused_sample = np.mean(smoothed_sample, axis=0)

        # smooth fused signal and scale to -1, 1 if necessary
        smoothed_fusion = smooth_fusion(fused_sample, args.post_smoothing_window)
        if args.std_annos_per_sample:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            output = scaler.fit_transform(smoothed_fusion)
        else:
            output = smoothed_fusion
        output = np.array(output).flatten()

        if args.plot:  # plot gold standard signal
            plt.figure()
            plt.plot(df[ts].values, output)
            plot_name = sample_id.split('.')[0] + '.png'
            plt.savefig(os.path.join(plot_path_method_dim, plot_name))
            plt.close()

        df = pd.DataFrame(data={'timestamp': df[ts].values, 'value': output})
        # df['timestamp'] = df['timestamp'].map(lambda x: '%.2f' % x)  # uncomment if needed
        # df['value'] = df['value'].map(lambda x: '%.4f' % x)  # uncomment if needed

        df.to_csv(os.path.join(save_path), index=False)  # save gold standard annotation

        print(f'Fused with {args.fusion}: {sample_id}')

        compute_interrater_agreement([np.expand_dims(np.array(smoothed_sample), 2)], interrater)

    if len(interrater) > 0:
        print('\nMean interrater agreement for the fused annotations: MEAN {0:3f} STD {1:3f}'.format(
            np.mean(interrater), np.std(interrater)))
