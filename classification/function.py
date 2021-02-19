"""
@author : haoran
time 2/8
"""

import torch.nn as nn
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score
import numpy as np
import torch

def wce_loss(pred,gt):
    # TODO 1 制作mask
    # print(pred.shape)


    # print(gt.shape)
    pred = pred
    # gt = gt.long()
    return nn.MultiLabelSoftMarginLoss()(pred,gt)
def my_acc_score(prediction,label):
    y = prediction.reshape(-1)
    l = label.reshape(-1)
    y = np.array(y.cpu().detach())
    y = np.where(y > 0.5, 1, 0).astype('int')
    l = np.array(l.cpu().detach()).astype('int')
    return accuracy_score(y,l)

def my_precision_score(prediction,label):
    y = prediction.reshape(-1)
    l = label.reshape(-1)
    y = np.array(y.cpu().detach())
    y = np.where(y > 0.5, 1, 0).astype('int')
    l = np.array(l.cpu().detach()).astype('int')
    return precision_score(y, l, zero_division=1)


def my_f1_score(prediction,label):

    y = prediction.reshape(-1)
    l = label.reshape(-1)

    y = np.array(y.cpu().detach())
    y = np.where(y > 0.5, 1, 0).astype('int')
    l = np.array(l.cpu().detach()).astype('int')

    return f1_score(y,l,zero_division=1)

def my_recall_score(prediction,label):

    y = prediction.reshape(-1)
    l = label.reshape(-1)
    y = np.array(y.cpu().detach())
    y = np.where(y > 0.5, 1, 0).astype('int')
    l = np.array(l.cpu().detach()).astype('int')

    return recall_score(y, l, zero_division=1)


if __name__ == '__main__':
    y = torch.zeros([1,23])
    l = torch.zeros((1,23))
    for i in range(5):
        l[0][i] = 1
    print(my_acc_score(y,l))