from sklearn.mixture import GaussianMixture


def run_GMM(data_to_fit, data_to_predict, k, seed=1):
    """Apply the GaussianMixtureModel on a dataset.
    For more see: https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
    Args:
        data_to_fit (int): The dataset
        data_to_predict (int): The dataset for prediction
        k (int): The number of components.
        seed (int): Random seed for initialisation

    Returns:
        [type]: [description]
    """
    gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=seed)
    labels = gmm.fit_predict(data_to_fit)
    if data_to_predict is None:
        preds = None
    else:
        preds = gmm.predict(data_to_predict)
    return labels, preds
