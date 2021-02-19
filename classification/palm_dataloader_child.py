"""
@author haoran
time:2021/2/7

"""
import traceback
import torch
import torchsummary
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import Dataset
import os
import random
from utils import father_cls_dict,child_cls_dict
from sklearn.model_selection import train_test_split
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt
from tensorboardX import writer

# TODO 1 DataLoader


class PalmData(Dataset):
    def __init__(self, data_dir=None, train_val_percent=0.2,train_mode='train'):
        self.dont_need_father_cls = [2,6,9]
        self.dont_need_child_cls = [2,6,9,81,85,34]
        self.father_cls_dict = father_cls_dict(self.dont_need_father_cls)
        self.child_cls_dict = child_cls_dict(self.dont_need_child_cls)
        if data_dir != None:
            self.data_dir = data_dir
        else:
            self.data_dir = '../../after_correction_rename'


        data_list = os.listdir(self.data_dir)
        X_train, X_test = train_test_split(
            data_list, test_size=train_val_percent, random_state=321)


        if train_mode == 'train':

            print('Train DataLoader')
            data_list = X_train
        else:

            print('Test DataLoader')
            data_list = X_test
        check_list = []
        self.error_list = []
        for idx, item in enumerate(data_list):
            try:
                self.parse_image_name(item)
                check_list.append(item)
            except Exception as e:
                print(e)
                self.error_list.append(item)

        print('The Error img is:', self.error_list)
        print(len(self.error_list))

        self.data_list = check_list
        self.transform_enhancement = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize((0.57,0.57,0.57), (0.18,0.18,0.18)),

        ])

        self.transform_rgb = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),

            # torchvision.transforms.Normalize((0.57, 0.57, 0.57), (0.18, 0.18, 0.18)),

        ])
        self.transform_prewitt = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),

            # torchvision.transforms.Normalize((0.57, 0.57, 0.57), (0.18, 0.18, 0.18)),

        ])

    def __getitem__(self, idx):
        rgb = Image.open(os.path.join(self.data_dir, self.data_list[idx]))
        if rgb.size != (512,512):
            print(self.data_list[idx])
        rgb = rgb.resize((512,512))
        # TODO 转化成3通道
        if len(rgb.split()) !=3:
            if len(rgb.split()) ==4:
                rgb = rgb.convert('RGB')
            elif len(rgb.split()) == 1:
                self.error_list.append(self.data_list[idx])
        else:
            pass


        parse_result = self.parse_image_name(self.data_list[idx])
        father_cls = parse_result['father_cls']
        child_cls = parse_result['child_cls']
        father_cls_label = torch.zeros([len(self.father_cls_dict)])
        child_cls_label = torch.zeros([len(self.child_cls_dict)])

        for i in range(len(self.father_cls_dict)):
            if i+1 in father_cls:
                father_cls_label[i] = 1

        for i in range(len(self.child_cls_dict)):
            if i+1 in child_cls:
                child_cls_label[i] = 1
        # print(label)
        rgb = self.transform_rgb(rgb)
        return {'img': rgb,
                'father_cls': father_cls_label,
                'child_cls':child_cls_label}

    def __len__(self):
        return len(self.data_list)

    def parse_image_name(self, name):
        """
        解析图片的名称，获取大类和小类标签
        :param name:图片的名称
        :return:
        """
        name_name = name.split(';')[0]
        _ = name.split(';')[-1]
        name_format = _.split('-')[-1]
        _cls = _.split('-')[:-1]
        cls = []
        for idx, i in enumerate(_cls):
            try:
                # TODO 剔除掉不需要的类别
                _ = (int(i.split(',')[0]), int(i.split(',')[1]))
                if _[0] not in self.dont_need_father_cls:
                    if int(str(_[0])+str(_[1])) not in self.dont_need_child_cls:
                        cls.append(_)
                else:
                    pass

            except:
                print('the error is ', cls)

        father_cls = []
        child_cls = []
        for idx, item in enumerate(cls):
            try:
                father_cls.append(self.father_cls_dict[str(int(item[0]))])
                # print('the :',str(int(item[0])) + str(int(item[1])))
                child_cls.append(self.child_cls_dict[str(int(item[0])) + str(int(item[1]))])
            except Exception as e:

                traceback.print_exc(e)


        # TODO 去重
        father_cls = list(set(father_cls))
        child_cls = list(set(child_cls))
        # # 我只需要1 3 4 5 7 8
        # other_flag = False
        # father_cls = list(set(father_cls))
        # # print(father_cls)
        # if 2 in father_cls:
        #     father_cls.remove(2)
        #     other_flag =True
        # if 6 in father_cls:
        #     father_cls.remove(6)
        #     other_flag = True
        # if 9 in father_cls:
        #     father_cls.remove(9)
        #     other_flag = True
        # if 10 in father_cls:
        #     father_cls.remove(10)
        #     other_flag = True
        #
        # for i in range(len(father_cls)):
        #     if father_cls[i] > 1 :
        #         father_cls[i] = father_cls[i] - 1
        #     if father_cls[i] > 6 :
        #         father_cls[i] = father_cls[i] - 1
        # # if other_flag:
        # #     father_cls.append(8)
        return {'name': name_name,
                'father_cls': father_cls,
                'child_cls': child_cls,
                'format': name_format,
               }

if __name__ == '__main__':
    train_dataset = PalmData(data_dir='../../after_correction_rename')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=1)
    for idx,item in enumerate(trainDataLoader):
        print(item['img'].shape)
        print(item['father_cls'].shape)
        print(item['child_cls'].shape)

        break
