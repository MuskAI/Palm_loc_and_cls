"""
删除部分小类的训练
"""

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
from palm_dataloader_child import PalmData
import os
from tqdm import tqdm
from torchvision import transforms
from model_child_cls_plus2 import PalmNet
from tensorboardX import SummaryWriter
from function import my_acc_score, my_f1_score, my_precision_score, my_recall_score
import traceback

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]
writer = SummaryWriter('./runs/' + '0219palm_cls_loss_record20个小类plus')

sys.path.append('./')


def train():
    # TODO 0 超参数区域
    batch_size = 6
    lr = 1e-2

    # ###############################

    # TODO 1 构建模型 数据加载 损失函数 优化器
    model = PalmNet().cuda()

    # writer.add_graph(model, (torch.ones(1,3,512,512).cuda()))
    palm_data_train = PalmData()
    trainDataLoader = torch.utils.data.DataLoader(palm_data_train, batch_size=batch_size, num_workers=3)

    # optimizer = optim.Adam(params=model.parameters(),lr=lr)

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20,40,60,80], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # TODO 1 构建模型 数据加载 损失函数
    # if not os.path.exists(model_name):
    #     traceback.print_exc('Please choose a right model path!!')
    # else:
    #     checkpoint = torch.load(model_name, map_location='cpu')
    #     model.load_state_dict(checkpoint['state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     optimizer.load_state_dict(checkpoint['opt_state_dict'])
    #     print('==>loaded model:', model_name)

    num_epochs = 200
    min_loss = 99999
    for epoch in range(0, num_epochs):
        scheduler.step(epoch)
        loss_avg = 0
        acc_avg = 0
        f1_avg = 0
        precision_avg = 0
        recall_avg = 0
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
            label = data['child_cls']
            # print(type(label))
            image, label = image.to(device), label.to(device)

            pred = model(image)
            loss = wce_loss(pred, label)
            acc = my_acc_score(pred, label)
            f1 = my_f1_score(pred, label)
            precision = my_precision_score(pred, label)
            recall = my_recall_score(pred, label)
            writer.add_scalar('loss', loss.item(), global_step=epoch * len(trainDataLoader) + i)
            writer.add_scalar('acc', acc, global_step=epoch * len(trainDataLoader) + i)

            writer.add_scalar('f1', f1, global_step=epoch * len(trainDataLoader) + i)
            writer.add_scalar('precision', precision, global_step=epoch * len(trainDataLoader) + i)

            writer.add_scalar('recall', recall, global_step=epoch * len(trainDataLoader) + i)

            print('epoch [%d],[%d / %d] The Loss:[%.6f], Acc:[%.6f]'
                  'F1:[%.6f] RECALL:[%.6f] PRECISION:[%.6f] Lr:[%.6f]  ' %
                  (epoch, i, len(trainDataLoader), loss.item(), acc, f1, recall, precision, scheduler.get_lr()[0]))
            loss.backward()
            optimizer.step()
            loss_avg += loss.item()
            acc_avg += acc
            f1_avg +=f1
            precision_avg+=precision
            recall_avg+=recall

        writer.add_scalars('avg_loss_f1_acc_precision_recall', {'loos': loss_avg / len(trainDataLoader),
                                                                'f1': f1_avg / len(trainDataLoader),
                                                                'acc': acc_avg/len(trainDataLoader),
                                                                'precision': precision_avg/len(trainDataLoader),
                                                                'recall':recall_avg/len(trainDataLoader)},global_step=epoch)
        writer.add_scalar('lr',scheduler.get_lr()[0],global_step=epoch)

        checkpoint = {
            'state_dict': model.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if loss_avg/len(trainDataLoader) < min_loss:
            min_loss = loss_avg/len(trainDataLoader)
            torch.save(checkpoint,
                       './save_cls_model/min_0219_model_epoch_%d_%.6f.pt' % (epoch, loss_avg / len(trainDataLoader)))

        if epoch%10==0:
            torch.save(checkpoint,
                       './save_cls_model/0219model_epoch_%d_%.6f.pt' % (epoch, loss_avg / len(trainDataLoader)))


if __name__ == "__main__":
    model_name = './save_cls_model/0218model_epoch_47_0.288202.pt'
    train()
