import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

import diarisation.som as som


def apply_feature_reduction(data, params):
    if params.tsne is not None:
        data = _apply_tsne(data, params.tsne)

    if params.pca is not None:
        data = _apply_pca(data, params.pca)

    if params.som:
        if params.neurons is None:
            neurons = 5 * np.sqrt(len(data.index))
            # Find the next perfect square
            params.neurons = int(np.floor(np.sqrt(neurons)) + 1)

        print(f"SOM seed: {params.som_seed}")
        som_trained = som.run_som(data=data, x=params.neurons, y=params.neurons, sigma=params.sigma, alpha=params.alpha,
                          iteration=params.iter, initialization=params.weights, topology=params.topology,
                          seed=params.som_seed)
        # Som Evaluation
        # the new data are the weights of the neurons
        # data = Som.get_weights(som, neurons, len(data.columns), data.columns)
        selected_features, target_name = som.som_feature_selection(som_trained, params.neurons, len(data.columns),
                                                                   data.columns, 0, 0.04)
        print("Target variable: {}\nSelected features {}".format(target_name, selected_features))
        data = data.loc[:, selected_features]

    return data


def _apply_tsne(data, components=2):
    """Apply TSNE on a given data set.

    Args:
        data (pd.DataFrame): input dataset
        components (int, optional): number of resulting components. Defaults to 2.

    Returns:
        pd.DataFrame: the reduced dataset
    """
    #n_iter = 5000
    #perplexity = 40
    reduced_data = TSNE(n_components=components, random_state=301).fit_transform(data)
    reduced_data = pd.DataFrame(reduced_data)
    reduced_data = reduced_data.dropna()
    return reduced_data


def _apply_pca(data, components=2):
    """Apply PCA on a given dataset.

    Args:
        data (pd.DataFrame): Input dataset
        components (int, optional): number of resulting components. Defaults to 2.

    Returns:
        pd.DataFrame: reduced dataset
    """
    pca = PCA(n_components=components)
    reduced_data = pca.fit_transform(data)
    reduced_data = pd.DataFrame(reduced_data)
    reduced_data = reduced_data.dropna()
    min_max_scale(reduced_data)

    '''
    for i, component in enumerate(abs(pca.components_)):
        df = pd.Series(component, index=data.columns.values)
        w, h = 10, 1 + 0.2 * data.shape[1]
        fig = df.nlargest(data.shape[1]).plot(kind='barh', figsize=(w, h)).get_figure()
        plt.title(f'{i+1}. component of PCA')
        plt.tight_layout()

        plt.show()
    '''

    return reduced_data


# Scale data
def min_max_scale(data: pd.DataFrame):
    """Scales a given dataFrame to the interval -1, 1

    Args:
        data (pd.DataFrame): the dataFrame to be scaled
    """
    max = data.max
    min = data.min
    for column in data:
        max = np.amax(data[column].to_numpy())
        min = np.amin(data[column].to_numpy())
        data[column] = data[column].apply(lambda x: ((((x - min) / (max - min)) * 2) - 1))


def univariate_feature_selection(data, labels, feature_names):
    n_best = 46
    data = data.to_numpy()
    labels = labels.to_numpy().flatten()
    print(f"Shapes data: {data.shape}, labels: {labels.shape}")

    # Split dataset to select feature and evaluate the classifier
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, stratify=labels, random_state=0
    )

    plt.figure(1)
    plt.clf()
    X_indices = np.arange(data.shape[-1])

    # Univariate feature selection with F-test for feature scoring
    # We use the default selection function to select the four
    # most significant features
    selector = SelectKBest(f_classif, k=n_best)
    selector.fit(X_train, y_train)
    p_values = np.add(selector.pvalues_, 1e-8)
    print(f"Selector pvalues: {p_values}")
    scores = -np.log10(p_values)
    #print(f"Selector pvalues: {selector.pvalues_}")
    #scores = -np.log10(selector.pvalues_) / 10
    print(f"Scores: {scores}")
    print(f"Scores MAX: {scores.max()}")
    scores /= scores.max()
    print(f"Scores: {scores}")
    plt.bar(X_indices - .45, scores, width=.2,
            label=r'Univariate score ($-Log(p_{value})$)')

    # #############################################################################
    # Compare to the weights of an SVM
    clf = make_pipeline(MinMaxScaler(), LinearSVC())
    clf.fit(X_train, y_train)
    print('Classification accuracy without selecting features: {:.3f}'.format(clf.score(X_test, y_test)))

    svm_weights = np.abs(clf[-1].coef_).sum(axis=0)
    svm_weights /= svm_weights.sum()

    plt.bar(X_indices - .25, svm_weights, width=.2, label='SVM weight')

    clf_selected = make_pipeline(
        SelectKBest(f_classif, k=n_best), MinMaxScaler(), LinearSVC()
    )
    clf_selected.fit(X_train, y_train)
    print('Classification accuracy after univariate feature selection: {:.3f}'
          .format(clf_selected.score(X_test, y_test)))

    svm_weights_selected = np.abs(clf_selected[-1].coef_).sum(axis=0)
    svm_weights_selected /= svm_weights_selected.sum()

    plt.bar(X_indices[selector.get_support()] - .05, svm_weights_selected,
            width=.2, label='SVM weights after selection')

    plt.title("Comparing feature selection")
    plt.xlabel('Feature number')
    plt.yticks(())
    plt.axis('tight')
    plt.legend(loc='upper right')
    plt.show()

    df = pd.Series(scores, index=feature_names)
    w, h = 10, 1 + 0.2 * scores.shape[0]
    fig = df.nlargest(scores.shape[0]).plot(kind='barh', figsize=(w, h)).get_figure()
    plt.title(f'Univariate scores (sorted)')
    plt.tight_layout()

    plt.show()
