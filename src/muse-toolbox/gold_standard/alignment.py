import pandas as pd
import glob
import os

from gold_standard.CTW import compute_CTW_alignment
from gold_standard.utils import make_output_dirs_align, get_annotator_mapping
from gold_standard.preprocessing import preprocess_signal, get_scalers


def align(args):
    # make dirs
    output_path_method, output_path_method_dim = make_output_dirs_align(args.output_path, args.alignment,
                                                                        args.dimension)
    data_dir = os.path.join(args.input_path, args.dimension) + '/*.csv'
    ts = args.ts  # e.g., 'timeStamp'

    if args.std_annos_all_samples:  # fit scalers to data
        mapping = get_annotator_mapping(args.anno_mapping_file, args.annotators)
        std_scalers, minmax_scalers = get_scalers(data_dir, ts, args, mapping)

    data_dirs = glob.glob(data_dir)
    args.end = len(data_dirs) if args.end == -1 else args.end
    for path in data_dirs[args.start:args.end]:  # iterate over all/part of the instances that are to be aligned
        path = path.replace("\\", '/')
        sample_id = path.split('/')[-1]
        save_path = os.path.join(output_path_method_dim, sample_id)

        # check if fusion has already been conducted for this data instance
        if os.path.isfile(save_path):
            continue

        # read data (sequences of different lengths and thus NaNs are allowed)
        df = pd.read_csv(path)

        if args.ts_path is not None:  # if sequences of different lengths are to be aligned the timestamps of the aligned sequences need to be given
            ts_path = os.path.join(args.ts_path, sample_id + '.csv')
            df_ts = pd.read_csv(ts_path)

        # preprocessing: smoothing + scaling
        if args.std_annos_all_samples:
            smoothed_sample, annotators = preprocess_signal(args.pre_smoothing, args.pre_smoothing_window,
                                                            args.std_annos_per_sample, args.std_annos_all_samples, df,
                                                            ts, mapping, std_scalers, minmax_scalers)
        else:
            smoothed_sample, annotators = preprocess_signal(args.pre_smoothing, args.pre_smoothing_window,
                                                            args.std_annos_per_sample, args.std_annos_all_samples, df,
                                                            ts)

        if args.alignment == 'ctw':
            aligned_sequences = compute_CTW_alignment(smoothed_sample, annotators)
            if args.ts_path is not None:
                aligned_sequences[ts] = df_ts[ts].values
            else:
                aligned_sequences[ts] = df[ts].values
            # df[ts] = df[ts].map(lambda x: '%.2f' % x)  # uncomment if needed

            aligned_sequences.to_csv(save_path, index=False)  # save aligned annotations

            print(f'Aligned with {args.alignment}: {sample_id}')

    return output_path_method
