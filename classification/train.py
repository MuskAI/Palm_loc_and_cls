"""
@author haoran
time : 2/8
"""

import random
import numpy as np
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
import torch.optim as optim
from function import wce_loss
from palm_dataloader import PalmData
import os
from tqdm import tqdm
from torchvision import transforms
from model_0217 import PalmNet
from tensorboardX import SummaryWriter
from function import my_acc_score
import traceback
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0,1]
writer = SummaryWriter('./runs/'+'0217palm_cls_loss_record3')

sys.path.append('./')
def train():
    # TODO 0 超参数区域
    batch_size = 25
    lr = 1e-2

    # ###############################


    # TODO 1 构建模型 数据加载 损失函数 优化器
    model = PalmNet().cuda()


    # writer.add_graph(model, (torch.ones(1,3,512,512).cuda()))
    palm_data_train = PalmData()
    palm_data_test = PalmData(train_mode='test')
    trainDataLoader = torch.utils.data.DataLoader(palm_data_train, batch_size=batch_size, num_workers=1)

    testDataLoader = torch.utils.data.DataLoader(palm_data_test, batch_size=batch_size, num_workers=1)

    # optimizer = optim.Adam(params=model.parameters(),lr=lr)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,35,50,60], gamma=0.1)
    # TODO 1 构建模型 数据加载 损失函数
    if not os.path.exists(model_name):
        traceback.print_exc('Please choose a right model path!!')
    else:
        checkpoint = torch.load(model_name, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['opt_state_dict'])
        print('==>loaded model:', model_name)

    num_epochs = 70

    for epoch in range(start_epoch+1,num_epochs):
        scheduler.step(epoch)
        loss_avg = 0
        acc_avg = 0
        for i, (name, param) in enumerate(model.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name, param, epoch)


        for i, data in enumerate(trainDataLoader):
            # print(data['img'].shape)
            # print(data['father_cls'].shape)
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            image = data['img']
            # print(type(image))
            label = data['father_cls']
            # print(type(label))
            image, label = image.to(device), label.to(device)

            pred = model(image)
            loss = wce_loss(pred, label)
            acc = my_acc_score(pred,label)
            writer.add_scalar('loss',loss.item(),global_step=epoch*len(trainDataLoader)+i)
            writer.add_scalar('acc', acc,global_step=epoch*len(trainDataLoader)+i)

            print('epoch [%d],[%d / %d] The Loss:[%.6f], Acc:[%.6f]Lr:[%.6f] ' %
                  (epoch,i,len(trainDataLoader), loss.item(), acc, scheduler.get_lr()[0]))
            loss.backward()
            optimizer.step()
            loss_avg += loss.item()
            acc_avg += acc

        writer.add_scalar('loss_avg',loss_avg/len(trainDataLoader),global_step=epoch)
        writer.add_scalar('acc_avg',acc_avg/len(trainDataLoader),global_step=epoch)


        checkpoint = {
            'state_dict': model.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, './save_cls_model/!0217model_epoch_%d_%.6f.pt' % (epoch, loss_avg/len(trainDataLoader)))
if __name__ == "__main__":
    model_name = './save_cls_model/!0217model_epoch_31_0.045036.pt'
    train()