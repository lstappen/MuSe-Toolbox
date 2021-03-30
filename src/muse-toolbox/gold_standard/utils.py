import os
import json


def make_output_dirs_fusion(output_path, method, dim):
    # dir for results
    output_path_method = os.path.join(output_path, method)
    output_path_method_dim = os.path.join(output_path_method, dim)
    if not os.path.isdir(output_path_method_dim):
        os.makedirs(output_path_method_dim, exist_ok=True)

    # dir for result plots
    plot_path_method = os.path.join(output_path_method, 'plots')
    plot_path_method_dim = os.path.join(plot_path_method, dim)
    if not os.path.isdir(plot_path_method_dim):
        os.makedirs(plot_path_method_dim, exist_ok=True)

    return output_path_method_dim, plot_path_method_dim


def make_output_dirs_align(output_path, method, dim):
    # dir for results
    output_path_method = os.path.join(output_path, method, 'alignment')
    output_path_method_dim = os.path.join(output_path_method, dim)
    if not os.path.isdir(output_path_method_dim):
        os.makedirs(output_path_method_dim, exist_ok=True)
    return output_path_method, output_path_method_dim


def get_annotator_mapping(mapping_file, annotators):
    if mapping_file is 'none':
        mapping = {str(a): int(a) for a in annotators}
    else:
        with open(mapping_file) as json_file:
            mapping = json.load(json_file)
    return mapping
