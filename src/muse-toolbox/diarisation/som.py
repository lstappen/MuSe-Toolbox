from minisom import MiniSom
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colorbar
import pickle
from matplotlib.lines import Line2D


# todo: Docstrings

def run_som(data: pd.DataFrame, x, y, sigma, alpha, iteration, initialization, topology,
            neighborhood_function="gaussian",
            activation_distance="euclidean",
            seed=1):
    """Trains the SOM on the given data with the parameters

    Args:
        data (pd.DataFrame): Datset on which the som is trained
        x (int): x dimension of the SOM.
        y (int): y dimension of the SOM.
        sigma (float): Spread of the neighborhood function
        alpha (float): learning_rate
        iteration (int): Maximum number of iterations
        initialization (string): Initializes the weights of the SOM
        topology (string): Topology of the map.
        neighborhood_function (str, optional): Function that weights the neighborhood of a position in the map. Defaults to "gaussian".
        activation_distance (str, optional): Distance used to activate the map.. Defaults to "euclidean".
        seed (int, optional): random seed for weight initialisation

    Returns:
        MiniSom: Returns the SOM.
    """
    input_len = len(data.columns)
    print(f'SOM input length: {input_len}')
    data = data.to_numpy()
    # use the default decay function
    som = MiniSom(x=x, y=y, input_len=input_len, sigma=sigma, learning_rate=alpha,
                  neighborhood_function=neighborhood_function, topology=topology,
                  activation_distance=activation_distance, random_seed=seed)
    if initialization == "pca":
        # training process converges faster
        som.pca_weights_init(data)
    else:
        som.random_weights_init(data)
    som.train(data, iteration)

    return som


def get_weights(som: MiniSom, neurons, features, labels):
    """Returns the weights of the neural network in form of a dataframe.

    Args:
        som (MiniSom): The MiniSom form which the weights are extracted. 
        neurons (int): Number of neurons.
        features (int): Number of features. 
        labels (list): List of the labels.

    Returns:
        [type]: [description]
    """
    weights = som.get_weights()
    print(f'SOM raw weights shape: {weights.shape}')
    weights = weights.reshape(neurons * neurons, features)
    weights = pd.DataFrame(weights)
    weights.columns = labels
    return weights


def som_feature_selection(som: MiniSom, n_neurons, n_features, labels, target_index=0, a=0.04):
    """ Performs feature selection based on a self organised map trained with the desired variables

    Args:
        som (MiniSom): The MiniSom form which the weights are extracted
        n_neurons (int): Number of neurons
        n_features (int): Number of features
        labels (list or list-like): List of the labels
        target_index (int): The position of the target variable in W and labels
        a (float): An arbitrary parameter in which the selection depends, values between 0.03 and 0.06 work well

    Returns:
        selected_labels (list): list of strings, holds the names of the selected features in order of selection
        target_name (string): The name of the target variable so that user is sure he gave the correct input
    """

    labels = list(labels)
    weights = som.get_weights()
    print(f'SOM raw weights shape: {weights.shape}')
    weights = weights.reshape(n_neurons * n_neurons, n_features)
    target_name = labels[target_index]

    # add random noise features to act as a selection threshold criterion
    rand_feat = np.random.uniform(low=0, high=1,
                                  size=(weights.shape[0], weights.shape[1] - 1))  # create N -1 random features
    W_with_rand = np.concatenate((weights, rand_feat), axis=1)  # add them to the N regular ones
    W_normed = (W_with_rand - W_with_rand.min(0)) / W_with_rand.ptp(0)  # normalize each feature between 0 and 1

    target_feat = W_normed[:, target_index]  # column of target feature

    # Two conditions to check against a
    check_matrix1 = abs(np.vstack(target_feat) - W_normed)
    check_matrix2 = abs(np.vstack(target_feat) + W_normed - 1)
    S = np.logical_or(check_matrix1 <= a, check_matrix2 <= a).astype(int)  # applies "or" element-wise in two matrices

    S[:, target_index] = 0  # ignore the target feature so that it is not picked

    selected_labels = []
    while True:

        S2 = np.sum(S, axis=0)  # add all rows for each column (feature)

        if not np.any(S2 > 0):  # if all features add to 0 kill
            break

        selected_feature_index = np.argmax(S2)  # feature with the highest sum gets selected first

        if selected_feature_index > (S.shape[1] - (rand_feat.shape[1] + 1)):  # if random feature is selected kill
            break

        selected_labels.append(labels[selected_feature_index])

        # delete all rows where selected feature evaluates to 1, thus avoid selecting complementary features
        rows_to_delete = np.where(S[:, selected_feature_index] == 1)
        S[rows_to_delete, :] = 0

    # selected_labels = [label for i, label in enumerate(labels) if i in feature_indeces]
    return selected_labels, target_name


def evaluate_som(som: MiniSom, data: pd.DataFrame):
    # Evaluate the MiniSom
    data = data.to_numpy()
    # print quantization_error and topographic_error
    print(tabulate(tabular_data=[som.quantization_error(data), som.topographic_error(data)],
                   headers=["quantization_error", "topographic_error"]))


# deprecated
def plot_map(som: MiniSom, data: pd.DataFrame, marker):
    """ Plot the MiniSom. This assumes that the MiniSom has a hexagonal topology. This plot considers the distance between the neurons.

    Args:
        som (MiniSom): The MiniSom that should be ploted
        data (pd.DataFrame): The data associated with the MiniSom
        marker ([int]): Labels of each data point
    """
    fig = plt.figure(figsize=(15, 13))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    xx, yy = som.get_euclidean_coordinates()
    umatrix = som.distance_map()
    weights = som.get_weights()

    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            # conversion since the map is hexagonal
            wy = yy[(i, j)] * 2 / np.sqrt(3) * 3 / 4
            hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95 / np.sqrt(3),
                                 facecolor=cm.Blues(umatrix[i, j]), alpha=.4, edgecolor='gray')
            ax.add_patch(hex)

    xrange = np.arange(weights.shape[0])
    yrange = np.arange(weights.shape[1])
    plt.xticks(xrange - .5, xrange)
    plt.yticks(yrange * 2 / np.sqrt(3) * 3 / 4, yrange)
    size_x = 1
    size_y = 2 / np.sqrt(3) * 3 / 4
    ax.set_xlim(-3 / 2 * size_x, size_x * len(xrange))
    ax.set_ylim(-size_y, size_y * len(yrange))

    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    cb1 = colorbar.ColorbarBase(ax_cb, cmap=cm.Blues, orientation='vertical', alpha=.4)
    cb1.ax.get_yaxis().labelpad = 16
    cb1.ax.set_ylabel('distance from neurons in the neighbourhood', rotation=270, fontsize=16)
    plt.gcf().add_axes(ax_cb)
    plt.savefig("Som1.svg", format="svg")
    plt.show()


def generate_legend(color, marker):
    custom_lines = []
    for m in marker:
        custom_lines.append(Line2D([0], [0], color=color[m], lw=2, label=marker[m]))
    return custom_lines


# deprecated
def plot_map_2(som: MiniSom, data: pd.DataFrame, marker):
    """ Plot the MiniSom. This assumes that the MiniSom has a hexagonal topology. This plot considers the cluster that belongs to a neuron.

    Args:
        som (MiniSom): The MiniSom that should be ploted
        data (pd.DataFrame): The data associated with the MiniSom
        marker ([int]): Labels of each data point
    """
    fig = plt.figure(figsize=(15, 13))
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    weights = som.get_weights()
    # always use nipy_spectral to have a uniform colorspace
    color = cm.nipy_spectral(np.linspace(0, 1, len(np.unique(marker))))
    for index, row in data.iterrows():
        x = np.array(row)
        w = som.winner(x)
        wx, wy = som.convert_map_to_euclidean(w)
        # conversion since the map is hexagonal
        wy = wy * 2 / np.sqrt(3) * 3 / 4
        hex = RegularPolygon((wx, wy), numVertices=6, radius=.95 / np.sqrt(3),
                             facecolor=color[marker[index]], alpha=.4, edgecolor="gray")
        ax.add_patch(hex)

    xrange = np.arange(weights.shape[0])
    yrange = np.arange(weights.shape[1])
    plt.xticks(xrange - .5, xrange)
    plt.yticks(yrange * 2 / np.sqrt(3) * 3 / 4, yrange)
    size_x = 1
    size_y = 2 / np.sqrt(3) * 3 / 4
    ax.set_xlim(-3 / 2 * size_x, size_x * len(xrange))
    ax.set_ylim(-size_y, size_y * len(yrange))
    unique_marker = np.unique(marker)
    ax.legend(handles=generate_legend(color, unique_marker), fontsize=14, bbox_to_anchor=(1.08, 1))
    plt.savefig("Som2.svg", format="svg", bbox_inches='tight')
    plt.show()
