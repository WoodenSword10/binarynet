import torch
import torch.nn as nn
from torch.autograd import Function

# 自己定义的求导方式
class BinarizeF(Function):
    # 创建torch.autograd.Function类的一个子类
    # 必须是staticmethod
    @staticmethod
    # 第一个是cxt, 第二个是input
    # cxt在这里类似与self，cxt的属性可以在backward中调用
    # 自己定义的Function中的forward()方法，所有的Variable参数将会转成tensor！
    # 因此这里的input也是tensor，在传入forward之前，autograd engine会自动将Variable unpack成tensor
    def forward(cxt, input):
        # 新建一个和input相同尺寸的output，将相应input中大于0的位置置为1， 小于0的位置置为-1
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        # grad_output为反向传播上一级计算得到的梯度值
        grad_input = grad_output.clone()
        return grad_input

# aliases
# 使用apply方法对自定义的求导方法取别名
binarize = BinarizeF.apply
