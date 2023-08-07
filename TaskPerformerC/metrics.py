from sklearn.metrics import roc_auc_score, log_loss
import numpy as np

def evaluate_metrics(y_true, y_pred, metrics):
    result = dict()
    for metric in metrics:
        if metric in ['logloss', 'binary_crossentropy']:
            result[metric] = log_loss(y_true, y_pred, eps=1e-7)
        elif metric == 'AUC':
            result[metric] = roc_auc_score(y_true, y_pred)
    return result