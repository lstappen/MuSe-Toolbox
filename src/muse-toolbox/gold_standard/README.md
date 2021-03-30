## Create Gold Standard Annotations from several raw ratings.

Dependencies:

- Python 3.7

- Install dependencies in requirements.txt with
    `pip -r requirements.txt`

- For CTW you need to install Octave 5.2.0 and check that 
    `os.environ['OCTAVE_EXECUTABLE'] == 'C:/Octave/Octave-5.2.0/mingw64/bin/octave-cli.exe'`


### Fuse annotations

Example call:
`python gold_standard/main.py -inp data/raw_annotations_original/ -out output --fusion dba -dim valence --plot --std_annos --pre_smoothing none --pre_smoothing_window 5 --post_smoothing_window 15`

| Arg               | Type           | Description  | Options | Default | Example |
| -------------     |:-------------:| -----| ----------| ---- | ---- |
| -inp --input_path      | str | Path to directory that contains `dimension` folder. In the `dimension` folder lie the csv files (format described below), each containing the annotations for one instance.| | `data\` | |
| -out --output_path     | str | Path where the generated gold standard annotations are to be saved | | `output\` | |
| -dim --dimension    | str      | The name of the dimension that is to be aligned| | `arousal` | `valence` |
| -fuse --fusion   | str | The type of fusion | `mean`, `ewe`, `dba`, or `none` | `none` | |
| -align --alignment   | str | The type of alignment | `ctw` or `none` | `none` | |
| --std_annos_per_sample  | bool | Whether to standardize the annotators per data sample | True or False | False | |
| --std_annos_all_samples  | bool | Whether to standardize the annotators over all data samples | True or False | False | |
| --pre_smoothing  | str | The type of smoothing that is to be applied in the beginning (Savitzky-Golay or Moving average filter) | `savgol`, `avg`, or `none` | `savgol` | |
| --pre_smoothing_window  | int | The window size for the pre_smoothing filter (odd number, greater than 2) |  | 5 |  7, 9, 11 |
| --post_smoothing_window  | int | The window size for the smoothing that is conducted after the fusion (greater than 1) | | 10 | 10, 15, 25 |
| --plot  | bool | Plot the gold standard annotations | True or False | False | |
| --anno_mapping_file  | str | A mapping file in json format containing a mapping of annotator ids (only needed if there are several ids per annotator) |  | `none`|  `annotator_id_mapping.json` |
| --annotators  | nargs | List of annotator ids that are to be included in the gold standard | | | 0 1 2 3 4 (for 5 annotators with ids 0-4)|


Example call for CTW alignment + EWE fusion: \
`python gold_standard/main.py -inp data/raw_annotations_original/ -out output --std_annos_all_samples --alignment ctw --fusion ewe -dim valence --plot --pre_smoothing savgol --pre_smoothing_window 5 --post_smoothing_window 15`

Example input csv file format (one timestamp column, one column per annotator): \
- Unique annotators are identified by integer id.

```
timeStamp,1,2,3,4,7 \
0.25,-33.925,0.0,7.8,7.8,270.0 \
0.5,-23.65,0.0,7.8,7.8,352.5 \
0.75,0.0,0.0,7.8,7.8,368.65 \
1.0,0.0,0.0,7.8,7.8,253.4
```
Example annotator_mapping.json file:
- Define mapping annotator mapping so that key anno id yields to actual anno id.
```
{"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8, "9": 7, "10": 0, "11": 3}
```