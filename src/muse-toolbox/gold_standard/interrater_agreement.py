import numpy as np


def compute_interrater_agreement(annos, interrater):
    # annos:  (num_seqs, num_annotators, seq_len, 1)
    num_annos = annos[0].shape[0]

    for seq in annos:
        # Compute a weight for each annotation
        r = np.ones((num_annos, num_annos))

        for anno in range(num_annos):  # Compute correlation between all annotator pairs
            for anno_comp in range(num_annos):
                if anno != anno_comp:
                    r[anno, anno_comp] = np.corrcoef(seq[anno, :, 0], seq[anno_comp, :, 0])[0, 1]

        r_mean = np.zeros_like(r)
        r_mean = np.mean(r_mean, axis=1)

        for anno_0 in range(num_annos):  # Compute mean correlation for each annotator ignoring self-agreements
            for anno_1 in range(num_annos):
                if anno_0 != anno_1:
                    r_mean[anno_0] += r[anno_0, anno_1]
            r_mean[anno_0] /= num_annos - 1

        r_mean[np.isnan(r_mean)] = 0.
        inter_rater_agreement = np.mean(r_mean)
        interrater.append(inter_rater_agreement)
    return interrater
