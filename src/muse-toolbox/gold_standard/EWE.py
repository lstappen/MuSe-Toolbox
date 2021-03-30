import numpy as np


def CCC(X1, X2):
    # Compute Concordance Correlation Coefficient (CCC)
    x_mean = np.nanmean(X1)
    y_mean = np.nanmean(X2)
    x_var = 1.0 / (len(X1) - 1) * np.nansum((X1 - x_mean) ** 2)
    y_var = 1.0 / (len(X2) - 1) * np.nansum((X2 - y_mean) ** 2)

    covariance = np.nanmean((X1 - x_mean) * (X2 - y_mean))
    return round((2 * covariance) / (x_var + y_var + (x_mean - y_mean) ** 2), 4)


def compute_EWE(annos):
    # Fuse annotations using the Evaluator Weighted Estimator (EWE)
    # annos: (num_seqs, num_annotators, seq_len, 1)
    # returns: (num_seqs, seq_len, 1)

    num_annos = annos[0].shape[0]

    EWE = []

    for seq in annos:

        # Compute a weight for each annotation
        r = np.ones((num_annos, num_annos))

        # Note: The ones on the main diagonal should be kept
        for anno in range(num_annos):
            for anno_comp in range(num_annos):
                if anno != anno_comp:
                    r[anno, anno_comp] = CCC(seq[anno, :, 0], seq[anno_comp, :, 0])

        r_mean = np.zeros_like(r)
        r_mean = np.mean(r_mean, axis=1)

        for anno_0 in range(num_annos):
            for anno_1 in range(num_annos):
                if anno_0 != anno_1:
                    r_mean[anno_0] += r[anno_0, anno_1]
            r_mean[anno_0] /= num_annos - 1

        r = r_mean
        r[np.isnan(r)] = 0.
        r[r < 0] = 0.  # Important: Give all negatively correlated annotations zero weight!
        r_sum = np.nansum(r)
        r = r / r_sum

        # Apply weights to get the Evaluator Weighted Estimator
        seq_len = seq.shape[1]
        EWEseq = np.zeros(seq_len)

        for anno in range(num_annos):
            EWEseq = np.round(EWEseq + seq[anno, :, 0] * r[anno], 3)
        EWE.append(EWEseq)
    return EWE
