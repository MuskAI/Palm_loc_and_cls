import torch.nn as nn
import math
import json
import traceback
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


def child_cls_dict(dont_need_cls):
    """
    小类的字典，为每一个小类给一个编号
    :return: 一个字典
    """
    dont_need_child_cls = []
    dont_need_father_cls = []
    # TODO 大类与小类判断
    for i in dont_need_cls:
        if i / 10 > 1:
            # 两位数，小类编号
            dont_need_child_cls.append(i)
        elif i/10 < 1:
            dont_need_father_cls.append(i)
        else:
            traceback.print_exc('The do not need list error,please make sure that all number is between 0 and 99')
    print('the do not need father cls is :', dont_need_father_cls)
    print('the do not need child cls is :', dont_need_child_cls)
    child_dict = {'11':1,
                  '12':2,
                  '13':3,
                  '14':4,

                  '21':5,
                  '22':6,
                  '23':7,

                  '31':8,
                  '32':9,
                  '33':10,
                  '34':11,
                  '35':12,

                  '41':13,
                  '42':14,
                  '43':15,
                  '44':16,

                  '51':17,
                  '52':18,

                  '61':19,
                  '62':20,

                  '71':21,
                  '72':22,
                  '73':23,

                  '81':24,
                  '82':25,
                  '83':26,
                  '84':27,
                  '85':28,

                  '91':29,
                  '92':30,
                  '93':31,
                  '94':32,
                  '95':33,
                  '96':34,
                  '97':35,
                  '98':36}

    # TODO 找到要删除大类下面的全部小类
    for i in dont_need_father_cls:
        for j in child_dict.keys():
            if i == int(j[0]):
                dont_need_child_cls.append(int(j))
    dont_need_child_cls = list(set(dont_need_child_cls))

    # print(dont_need_child_cls)
    for i in dont_need_child_cls:
        child_dict.pop(str(i))
    for idx,item in enumerate(child_dict):
        child_dict[str(item)]=idx+1


    print('====>>>>>>>The following is the child cls you need=======>')
    print(json.dumps(child_dict, indent=4))
    print(len(child_dict))
    return child_dict

def father_cls_dict(dont_need_cls):
    """
    大类的字典，为每一个大类给一个编号
    :param dont_need_cls:[1,2,3] 不需要的类的编号
    :return:
    """
    father_dict = {'1':1,
                  '2':2,
                  '3':3,
                  '4':4,
                  '5':5,
                  '6':6,
                  '7':7,
                  '8':8,
                  '9':9}
    for i in dont_need_cls:
        father_dict.pop(str(i))
    for idx,item in enumerate(father_dict):
        father_dict[str(item)]=idx+1

    print('====>>>>>>>The following is the father cls you need=======>')
    print(json.dumps(father_dict,indent=4))
    return father_dict
if __name__ == '__main__':
    child_cls_dict([1,31,21,41,11])