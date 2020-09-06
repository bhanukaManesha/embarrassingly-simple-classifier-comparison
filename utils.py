
from sklearn.metrics import roc_curve, auc
from itertools import cycle, product
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import torch

import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 7.5, 10


def results(y_true, y_pred, classes):
    y_pred_ = y_pred.argmax(dim=1)

    #classification report
    print(classification_report(y_true, y_pred_, target_names=classes))

    # AUC curve
    y_true_ohe = np.zeros((len(y_pred), len(classes)))
    for idx, lbl in enumerate(y_true):
        y_true_ohe[idx][lbl] = 1

    y_pred = y_pred.detach().numpy()

    plot_multiclass_roc(y_true_ohe,y_pred, classes=classes)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_)
    plot_confusion_matrix(cm, classes)

def plot_multiclass_roc(y_true, y_pred, classes):
    n_classes = len(classes)
    lw=1
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['#eeefff', '#eaaffd', '#eaefaf'])
    for i, color in zip(range(n_classes), colors):
        if roc_auc[i] < 0.9:
            plt.plot(fpr[i], tpr[i], lw=lw,
                     label=f'ROC curve of  {classes[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()