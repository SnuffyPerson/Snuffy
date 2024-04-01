import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def get_labels(neg, pos):
    labels = np.concatenate((np.ones(pos.shape), np.zeros(neg.shape)))
    vec = np.concatenate((pos, neg))
    return labels, vec


def get_roc_sklearn(dtest, dood):
    labels, vec = get_labels(dtest, dood)
    return roc_auc_score(labels, vec)


def get_pr_sklearn(dtest, dood):
    labels, vec = get_labels(dtest, dood)
    return average_precision_score(labels, vec)


def get_fpr(dtest, dood):
    labels, vec = get_labels(dtest, dood)
    return roc_curve(labels, vec)[0]


def get_scores_one_cluster(ftrain, ftest, food):
    pass


def get_features():
    pass
