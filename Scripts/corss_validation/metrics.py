import numpy as np

def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(np.concatenate([y_true, y_pred]))
    matrix = np.zeros((len(classes), len(classes)), dtype=int)

    class_to_index = {c: i for i, c in enumerate(classes)}

    for t, p in zip(y_true, y_pred):
        i = class_to_index[t]
        j = class_to_index[p]
        matrix[i, j] += 1

    return matrix


def accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.mean(y_true == y_pred))


def error_rate(y_true, y_pred):
    return float(1 - accuracy(y_true, y_pred))


def precision_macro(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    scores = []

    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        scores.append(precision)

    return float(np.mean(scores))


def recall_macro(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    scores = []

    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        scores.append(recall)

    return float(np.mean(scores))


def specificity_macro(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    scores = []

    total = cm.sum()

    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        scores.append(specificity)

    return float(np.mean(scores))


def f1_macro(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    scores = []

    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        scores.append(f1)

    return float(np.mean(scores))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return float(np.mean(np.abs(y_true - y_pred)))


def compute_all_metrics(y_true, y_pred, y_proba=None):
    return {
        "accuracy": accuracy(y_true, y_pred),
        # "error_rate": error_rate(y_true, y_pred),
        "precision_macro": precision_macro(y_true, y_pred),
        "recall_macro": recall_macro(y_true, y_pred),
        # "specificity_macro": specificity_macro(y_true, y_pred),
        "f1_macro": f1_macro(y_true, y_pred),
        # "rmse": rmse(y_true, y_pred),
        # "mae": mae(y_true, y_pred),
    }
