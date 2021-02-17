"""
@author haoran
time : 2/17
测试程序
"""

import torch
import torch.nn as nn
import sys
from function import wce_loss
from palm_dataloader import PalmData
import os
from tqdm import tqdm
from torchvision import transforms
from model import PalmNet
import traceback
from tensorboardX import SummaryWriter
from function import my_acc_score

device = 'cuda'
sys.path.append('./')
sys.path.append('./save_cls_model')

def test():
    model = PalmNet().cuda()
    model.eval()
    # TODO 0 超参数区域
    lr = 1e-2


    # ###############################


    # TODO 1 构建模型 数据加载 损失函数
    if not os.path.exists(model_name):
        traceback.print_exc('Please choose a right model path!!')
    else:
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['state_dict'])
        print('==>loaded model:',model_name)

    palm_data_test = PalmData(train_mode='train')
    testDataLoader = torch.utils.data.DataLoader(palm_data_test, batch_size=1, num_workers=1)
    acc_avg = 0

    for i, data in enumerate(testDataLoader):
        image = data['img']
        # print(type(image))
        label = data['father_cls']
        # print(type(label))
        image, label = image.to(device), label.to(device)
        pred = model(image)

        loss = wce_loss(pred, label)
        acc = my_acc_score(pred,label)
        acc_avg +=acc
        print('When Testing [%d / %d] The Loss:[%.6f] The ACC:[%.6f]' % (i,len(testDataLoader), loss.item(), acc))
        print('The avg acc is :',acc_avg/(i+1))
        print('The pred: ', pred)
        print('The Gt: ', label)
        print('=================================\n')
if __name__ == "__main__":
    model_name = './save_cls_model/0216model_epoch_45_0.035881.pt'
    test()