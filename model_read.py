import torch
from torch import nn
from modules import BinaryLinear, BinaryTanh
import cv2
import numpy as np

# 输入尺寸
input_size = 784
# 隐藏层神经元个数
hidden_size = 1024
# 隐藏层层数
num_layers = 1
# 种类数
num_classes = 10
# 输入层随机置零概率
drop_in = 0.2
# 隐藏层随机置零概率
drop_hid = 0.5
# 归一化参数，
# momentum为移动平均的动量值，
momentum = 0.9
# eps为数值稳定性而加到分母上的值
eps = 1e-6

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(MLP, self).__init__()
        # 获取隐藏层层数
        self.num_layers = num_layers

        # 随机将输入张量中的部分元素设置为0， p为将元素置0的概率
        self.p_in = nn.Dropout(p=drop_in)
        # 设置每层网络
        for i in range(1, self.num_layers + 1):
            # 若i=1输入层，输入特征数为input_size，否则为隐藏层，输入特征为隐藏层神经元个数
            in_features = input_size if i == 1 else hidden_size
            # 输出结果数为隐藏层神经元个数
            out_features = hidden_size
            # 网络层功能设置
            layer = nn.Sequential(
                # 二值化线性层， 无偏置
                BinaryLinear(in_features, out_features, bias=False),
                # 对out_features维度进行归一化，momentum为移动平均的动量值，eps为数值稳定性而加到分母上的值
                nn.BatchNorm1d(out_features, momentum=momentum, eps=eps),
                # 激活函数
                BinaryTanh(),
                # 随机置零
                nn.Dropout(p=drop_hid))
            # 设置属性值，参数分别为对象、名称、属性值
            setattr(self, 'layer{}'.format(i), layer)
        # 输出层设置
        self.fc = BinaryLinear(hidden_size, num_classes, bias=False)

    def forward(self, x):
        # 将输入x随机置0
        out = self.p_in(x)
        # 依次执行每层操作
        for i in range(1, self.num_layers + 1):
            out = getattr(self, 'layer{}'.format(i))(out)
        # 执行输出层获取结果
        out = self.fc(out)
        return out

mymodel = MLP(input_size, hidden_size, num_classes, num_layers=num_layers)
mymodel.load_state_dict(torch.load('bnn1024.pkl'))

i = 0
for para in mymodel.parameters():
    print(para.data.shape)
    if i == 0:
        w1 = para.data.numpy()
    elif i == 3:
        w2 = para.data.numpy()
    i += 1

img = cv2.imread('final_4.bmp', 0)
img_Data = np.array(img)
img_Data = img_Data.reshape((1, 784))
w1 = np.where(w1>0, 1, -1)
w2 = np.where(w2>0, 1, -1)
print(w1)
a1 = np.dot(img_Data, w1.T)
print(a1)
a1 = np.where(a1>0, 1, -1)
a2 = np.dot(a1, w2.T)
print(a2)