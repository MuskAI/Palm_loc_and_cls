"""
@author : haoran
time 2/8
"""

import torch.nn as nn
from sklearn.metrics import accuracy_score
import numpy as np

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