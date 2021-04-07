#!/bin/sh

export PYTHONUNBUFFERED=1

# Fusing several annotations to one Gold Standard using different methods

input=examples/raw  # path to input raw annotations

emotion_dim=arousal  # emotion dimension which is to be fused, annotations must be in 'arousal' folder
pre_smooth_filter=savgol  # smooth the individual annotations with a Savitzky-Golay filter
pre_smoothing_window=5  # the window size of the Savitzky-Golay filter
post_smoothing_window=15  # smoothing window of the filter used on the fused annotation
annotators="1 2 3"  # annotator ids whose annotations are to be fused

######################

output_unaligned="output/std_all_samples_${pre_smooth_filter}_${pre_smoothing_window}_${post_smoothing_window}_unaligned"  # path where the fused annotations are to be saved

# 1. Fusion using mean
fusion=mean
python src/muse-toolbox gold_standard -inp $input -out $output_unaligned --std_annos_all_samples --fusion $fusion -dim $emotion_dim  --pre_smoothing $pre_smooth_filter --pre_smoothing_window $pre_smoothing_window --post_smoothing_window $post_smoothing_window --annotators $annotators --ts timestamp --plot &>> out.txt

# 2. Fusion using EWE
fusion=ewe
python src/muse-toolbox gold_standard -inp $input -out $output_unaligned --std_annos_all_samples --fusion $fusion -dim $emotion_dim  --pre_smoothing $pre_smooth_filter --pre_smoothing_window $pre_smoothing_window --post_smoothing_window $post_smoothing_window --annotators $annotators --ts timestamp --plot &>> out.txt

# 3. Fusion using DBA
fusion=dba
python src/muse-toolbox gold_standard -inp $input -out $output_unaligned --std_annos_all_samples --fusion $fusion -dim $emotion_dim  --pre_smoothing $pre_smooth_filter --pre_smoothing_window $pre_smoothing_window --post_smoothing_window $post_smoothing_window --annotators $annotators --ts timestamp --plot &>> out.txt

######################

output_aligned="output/std_all_samples_${pre_smooth_filter}_${pre_smoothing_window}_${post_smoothing_window}_aligned"  # path where the fused annotations are to be saved

# 4. Alignment with CTW and fusion using EWE (RAAW)
fusion=ewe
alignment=ctw
python src/muse-toolbox gold_standard -inp $input -out $output_aligned --std_annos_all_samples --alignment $alignment --fusion $fusion -dim $emotion_dim  --pre_smoothing $pre_smooth_filter --pre_smoothing_window $pre_smoothing_window --post_smoothing_window $post_smoothing_window --annotators $annotators --ts timestamp --plot &>> out.txt


######################

# Plot fusions
fusion_output_paths="$output_unaligned $output_aligned"
python scripts/plot_gs.py --paths $fusion_output_paths &>> out.txt