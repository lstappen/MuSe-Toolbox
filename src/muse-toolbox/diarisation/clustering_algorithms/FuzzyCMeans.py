import numpy as np
import skfuzzy as fuzz


def run_fuzzy_cmeans(data_to_fit, data_to_predict, k, m, seed=1):
    """Apply FuzzyCMeans on a dataset. 
    For more see: https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_cmeans.html

    Args:
        data_to_fit (pd.DataFrame): The dataset
        data_to_predict (pd.DataFrame): The dataset for prediction
        k (int): The number of clusters
        m (int): The fuzzifier m.
        seed (int, optional): random seed for initialisation (default: 1)

    Returns:
        [int], float: The clusters for each point and the fpc score
    """
    data_to_fit = data_to_fit.to_numpy()
    data_to_fit = data_to_fit.T
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data_to_fit, k, 2, error=0.0005, maxiter=10000, init=None, seed=seed)
    labels = np.argmax(u, axis=0)

    if data_to_predict is None:
        preds = None
    else:
        data_to_predict = data_to_predict.to_numpy()
        data_to_predict = data_to_predict.T
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(data_to_predict, cntr, 2, error=0.0005, maxiter=10000,
                                                                 init=None, seed=seed)
        preds = np.argmax(u, axis=0)
    # TODO merge fpc
    return labels, preds, fpc
