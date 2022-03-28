import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import *


class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        # Hardtanh就是1个线性分段函数[-1, 1]，当x>1时将x置为1，当x<-1时将x置为-1，其余保持
        self.hardtanh = nn.Hardtanh()

    def forward(self, input):
        # 对input执行hardtanh激活函数
        output = self.hardtanh(input)
        # 求导函数，这里仅将output2值化，大于0的置为1，小于0的置为-1
        output = binarize(output)
        return output
        

class BinaryLinear(nn.Linear):
    # 二值化线性层
    def forward(self, input):
        # 将权重二值化
        binary_weight = binarize(self.weight)
        # 判断是否存在偏置
        if self.bias is None:
            return F.linear(input, binary_weight)
        else:
            return F.linear(input, binary_weight, self.bias)

    # 初始化参数
    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        # 返回数的平方根
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv



class BinaryConv2d(nn.Conv2d):

    def forward(self, input):
        bw = binarize(self.weight)
        return F.conv2d(input, bw, self.bias, self.stride,
                               self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        # Glorot initialization
        in_features = self.in_channels
        out_features = self.out_channels
        for k in self.kernel_size:
            in_features *= k
            out_features *= k
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv
