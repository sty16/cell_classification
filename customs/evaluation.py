import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def get_class_accuracy(pred, target, classes, topk=1, thrs=0.):
    pred_labels = np.argmax(pred, axis=1)
    gt_labels = target
    res_precision = precision_score(gt_labels, pred_labels, average=None)
    res_recall = recall_score(gt_labels, pred_labels, average=None)
    res_recall = [round(x, 3) for x in res_recall]
    res_precision = [round(x, 3) for x in res_precision]
    eval_result = {}
    assert len(res_recall) == len(classes)
    # eval_result['f1_score'] = {k: v for k, v in zip(classes, res_f1_score)}
    eval_result['precision'] = {k: v for k, v in zip(classes, res_precision)}
    eval_result['recall   '] = {k: v for k, v in zip(classes, res_recall)}
    print("\n")
    for metrics, value in eval_result.items():
        print(metrics, value)
    return eval_result
