import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def summarise(y_true, prob, thr=0.5):
    y_true = np.asarray(y_true); pred = (np.asarray(prob) > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sen = tp / (tp + fn) if tp + fn else float("nan")
    spe = tn / (tn + fp) if tn + fp else float("nan")
    pre = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * pre * sen / (pre + sen) if (pre + sen) else float("nan")
    auc = roc_auc_score(y_true, prob) if len(set(y_true)) == 2 else float("nan")
    return {"accuracy": (tp + tn) / len(y_true), "sensitivity": sen, "specificity": spe,
            "precision": pre, "f1": f1, "auc": auc, "TP": int(tp), "TN": int(tn),
            "FP": int(fp), "FN": int(fn), "n": int(len(y_true))}
