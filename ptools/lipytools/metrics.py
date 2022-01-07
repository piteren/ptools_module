from sklearn.metrics import precision_recall_fscore_support

# wrap for sklearn.precision_recall_fscore_support
def prf(val_labels, val_preds) -> tuple:
    pc, rc, f1, _ = precision_recall_fscore_support(
        val_labels,
        val_preds,
        average=        'macro',
        zero_division=  0)
    return pc, rc, f1