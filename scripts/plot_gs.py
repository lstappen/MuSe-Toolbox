import matplotlib.pyplot as plt
import pandas as pd
import os
from os import listdir
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--paths', nargs='+')
params = parser.parse_args()

data_path = 'examples/raw/arousal'

for file in listdir(data_path):
    emo = data_path.split('/')[-1]
    df = pd.read_csv(os.path.join(data_path, file))

    timestamps = df['timestamp'].values

    plt.figure()
    # plot raw annotations
    for anno in ['1', '2', '3']:
        plt.plot(timestamps, df[anno].values, color="gray")
    # plot fused annotations
    for path in params.paths:
        for fusion in glob.glob(path + '*/*'):
            fusion = fusion.replace("\\", '/')
            print(fusion)
            if 'unaligned' not in fusion and 'ctw' in fusion:  # no fusion only aligned annotations
                continue
            elif 'unaligned' not in fusion and 'ewe' in fusion:
                print('raaw')
                name = 'raaw'
            elif 'unaligned' not in fusion and 'mean' in fusion:
                name = 'ctw+mean'
            else:
                name = fusion.split('/')[-1]
            fusion_df = pd.read_csv(os.path.join(fusion, emo, file))
            plt.plot(timestamps, fusion_df['value'].values, label=name.upper(), linewidth=2.0)

    plt.xlabel('time (s)')
    plt.legend(loc="upper right")
    plt.ylim([-1, 1])

    # plt.show()
    plt.savefig(os.path.join('output', file.split('.')[0] + '.png'))

    plt.close()
