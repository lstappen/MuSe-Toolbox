import pandas as pd
import numpy as np


def compute_CTW_alignment(data, annotators):
    warping_paths = compute_CTW_warping_paths(data)
    aligned_data = {a: [] for a in annotators}
    for i, (a, path) in enumerate(zip(annotators, warping_paths)):  # iterate over warping paths per annotator
        for j in path:
            aligned_data[a].append(data[i][j])
    aligned_df = pd.DataFrame(aligned_data)
    return aligned_df


def compute_CTW_warping_paths(data, factor=1.0):
    from oct2py import octave
    # os.environ['OCTAVE_EXECUTABLE'] = 'C:/Octave/Octave-5.2.0/mingw64/bin/octave-cli.exe'

    if not isinstance(data, np.ndarray) and np.all(len(data) == len(data[0])):
        data = np.stack(data, axis=0)  # shape: [num_seq, seq_len]

    octave.addpath(octave.genpath('gold_standard/ctw'))
    octave.eval('pkg load optim')

    out = octave.feval('applyctw', data, len(data), factor)

    warping_paths = out.P.T.round(0).astype(int) - 1  # simply round indicies
    return warping_paths
