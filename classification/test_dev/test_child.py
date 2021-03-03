"""
@author haoran
time : 2/23
用于部署的小类测试程序

"""

import torch
import numpy as np
import torchvision.transforms as T
import sys
from PIL import Image
import cv2 as cv
import os
from model_child_cls_plus import PalmNet
import traceback
import matplotlib.pylab as plt

device = 'cpu'

class Diagnose:
    def __init__(self, img, model_path, output_path):
        """
        用于部署时候的小类疾病诊断类
        :param img: 输入是cv MAT
        :param model_path: 模型参数所在路径
        :param output_path: 诊断结果保存路径

        """

        self.model_path = model_path
        self.output_path = output_path

        # TODO let img be the (1,3,512,512) tensor
        if True:
            img = np.array(img,dtype='uint8')
            r = Image.fromarray(img[:, :, 2]).convert('L')
            g = Image.fromarray(img[:, :, 1]).convert('L')
            b = Image.fromarray(img[:, :, 0]).convert('L')
            img = Image.merge("RGB", (r, g, b))

        img = T.Compose([
            T.ToTensor()
        ])
        img = img.unsqueeze(0)
        pred_sigmoid = self.test(img)
        self.deal_with_output(pred_sigmoid)

    def test(self,img):

        model_name = self.model_path
        model = PalmNet().cpu()
        model.eval()
        data = img
        # TODO 1 构建模型 数据加载 损失函数
        if not os.path.exists(model_name):
            traceback.print_exc('Please choose a right model path!!')
        else:
            checkpoint = torch.load(model_name, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            print('==>loaded model:', model_name)
            image = data
            image = image.to(device)
            pred = model(image)
            pred_sigmoid = torch.nn.Sigmoid()(pred)
            print('The pred: ', np.where((pred.detach().numpy()) > 0.5, 1, 0))
            print('The pred_sigmoid: ', np.where((pred_sigmoid.detach().numpy()) > 0.5, 1, 0))
            print('=================================\n')

        return pred_sigmoid
    def deal_with_output(self,pred):
        child_cls_label = [1,2,3,4,5
                           ,6,7,8,9,10
                           ,11,12,13,14,15
                           ,16,17,18,19]
        plt.bar(child_cls_label,list(pred))
        plt.show()
        pass

