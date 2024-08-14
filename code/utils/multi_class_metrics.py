#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :multi_class_metrics.py
@Author :CodeCat
@Date   :2023/8/20 16:04
"""
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay


def get_confusion_matrix(trues, preds, labels=None):
    """
    获取混淆矩阵
    :param trues: 真实值 (n, )
    :param preds: 预测值 (n, )
    :param labels:
    :return:
    """
    if labels is None:
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    conf_matrix = confusion_matrix(
        y_true=trues,
        y_pred=preds,
        labels=labels
    )
    return conf_matrix


def plot_confusion_matrix(conf_matrix, class_indict):
    """
    绘制混淆矩阵
    :param conf_matrix: 混淆矩阵
    :param class_indict: 类别字典信息 eg: {'0': class1, '1': 'class2', ...}
    :return:
    """
    plt.imshow(conf_matrix)
    indices = range(conf_matrix.shape[0])
    labels = [class_indict[str(i)] for i in indices]
    # plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('Predictions')
    plt.ylabel('Trues')
    # 显示数据
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.show()


def model_metrics(model, dataloader, device, class_indict):
    """
    从 precision、accuracy、recall、f1 四个方面来评价模型
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        recall = tp / (tp + fn)
        f1 = 2 * precison * reacll / (precision + recall)
    """
    metrics_preds = []
    metrics_trues = []
    metrics_probs = []
    model.eval()
    with torch.no_grad():
        dataloader = tqdm(dataloader)
        for step, data in enumerate(dataloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            pred_classes = pred.argmax(dim=1)
            pred_probs = pred.softmax(dim=1)
            metrics_preds.extend(pred_classes.detach().cpu().numpy())
            metrics_probs.extend(pred_probs.detach().cpu().numpy())
            metrics_trues.extend(labels.detach().cpu().numpy())

    precision = precision_score(
        y_true=metrics_trues,
        y_pred=metrics_preds,
        average='micro'
    )
    accuracy = accuracy_score(
        y_true=metrics_trues,
        y_pred=metrics_preds
    )
    recall = recall_score(
        y_true=metrics_trues,
        y_pred=metrics_preds,
        average='micro'
    )
    f1 = f1_score(
        y_true=metrics_trues,
        y_pred=metrics_preds,
        average='micro'
    )
    print(classification_report(
        y_true=metrics_trues,
        y_pred=metrics_preds,
        target_names=[class_indict[str(i)] for i in range(len(class_indict))]
    ))
    conf_matrix = get_confusion_matrix(
        trues=metrics_trues,
        preds=metrics_preds
    )
    plot_confusion_matrix(conf_matrix, class_indict=class_indict)
    print("[Metrics] accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}".format(
        accuracy, precision, recall, f1
    ))
    return metrics_trues, metrics_preds, metrics_probs


def get_roc_pr(trues, preds, probs, labels=None):
    if labels is None:
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    trues = label_binarize(trues, classes=labels)
    preds = label_binarize(preds, classes=labels)
    probs = np.array(probs)
    fpr, tpr, roc_auc = dict(), dict(), dict()
    precision, recall, ap = dict(), dict(), dict()
    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(trues[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(trues[:, i], probs[:, i])
        ap[i] = average_precision_score(trues[:, i], probs[:, i])

    fpr['micro'], tpr['micro'], _ = roc_curve(trues.ravel(), preds.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    precision['micro'], recall['micro'], _ = precision_recall_curve(trues.ravel(), probs.ravel())
    ap['micro'] = average_precision_score(trues.ravel(), probs.ravel())

    return fpr, tpr, roc_auc, precision, recall, ap


def plot_roc(fpr, tpr, roc_auc, class_indict):
    """
    绘制ROC曲线
    """
    num_classes = len(class_indict)
    plt.figure()
    plt.plot(fpr['micro'], tpr['micro'], label='micro-average ROC curve (area = {0:0.3f})'.format(roc_auc["micro"]),
             linestyle=':', linewidth=4)
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of {0} (area = {1:0.3f})'.format(class_indict[str(i)], roc_auc[i]),
                 linewidth=2)

    plt.plot([0, 1], [0, 1], linestyle='-.', linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class ROC curve')
    plt.legend(loc='lower right')
    plt.show()


def plot_pr(precision, recall, ap, class_indict):
    """
    绘制PR曲线
    """
    num_classes = len(class_indict)
    plt.figure()
    plt.plot(precision['micro'], recall['micro'], drawstyle="steps-post", label='micro-average Precision-recal (AP = {0:0.3f})'.format(ap["micro"]),
             linestyle=':', linewidth=4)
    for i in range(num_classes):
        plt.plot(precision[i], recall[i], drawstyle="steps-post", label='Precision-recall for {0} (AP = {1:0.3f})'.format(class_indict[str(i)], ap[i]),
                 linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Multi-Class Precision-Recall curve')
    plt.legend(loc='lower left')
    plt.show()