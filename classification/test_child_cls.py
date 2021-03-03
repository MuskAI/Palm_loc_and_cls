"""
@author haoran
time : 2/17
测试程序
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from function import wce_loss
from palm_dataloader_child import PalmData
import os
from tqdm import tqdm
from torchvision import transforms
from model_child_cls_plus import PalmNet
import traceback
from tensorboardX import SummaryWriter
from function import my_acc_score,my_f1_score,my_recall_score,my_precision_score

device = 'cpu'
sys.path.append('./')
sys.path.append('./save_cls_model')

def test():
    model = PalmNet().cpu()
    model.eval()
    # TODO 0 超参数区域
    lr = 1e-2


    # ###############################


    # TODO 1 构建模型 数据加载 损失函数
    if not os.path.exists(model_name):
        traceback.print_exc('Please choose a right model path!!')
    else:
        checkpoint = torch.load(model_name,map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print('==>loaded model:',model_name)

    palm_data_test = PalmData(train_mode='te st')
    testDataLoader = torch.utils.data.DataLoader(palm_data_test, batch_size=1, num_workers=1)

    acc_avg = 0
    loss_avg = 0
    f1_avg = 0
    precision_avg = 0
    recall_avg = 0
    for i, data in enumerate(testDataLoader):
        image = data['img']
        # print(type(image))
        label = data['child_cls']
        # print(type(label))
        image, label = image.to(device), label.to(device)
        pred = model(image)

        loss = wce_loss(pred, label)
        acc = my_acc_score(pred, label)
        f1 = my_f1_score(pred, label)
        precision = my_precision_score(pred, label)
        recall = my_recall_score(pred, label)

        acc_avg += acc
        f1_avg += f1
        precision_avg += precision
        recall_avg += recall
        print('[%d / %d] The Loss:[%.6f], Acc:[%.6f]'
              'F1:[%.6f] RECALL:[%.6f] PRECISION:[%.6f]' %
              (i, len(testDataLoader), loss.item(), acc, f1, recall, precision))
        print('The avg Acc:[%.3f]'
              'F1:[%.3f] RECALL:[%.3f] PRECISION:[%.3f] is :'% (acc_avg/(i+1),f1_avg/(i+1),
                                                                recall_avg/(i+1),precision_avg/(i+1)))
        print('The pred: ', np.where((pred.detach().numpy())>0,1,0))
        print('The Gt: ', label.detach().numpy().astype('int'))
        print('=================================\n')
if __name__ == "__main__":
    model_name = './save_cls_model/0220model_epoch_93_0.148907.pt'
    test()