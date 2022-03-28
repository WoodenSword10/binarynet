import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

from modules import BinaryLinear, BinaryTanh
from adam import Adam

# 设置CPU生成随机数的种子，方便下次复现实验结果。
torch.manual_seed(1111)


# Hyper Parameters
# 输入尺寸
input_size = 784
# 隐藏层神经元个数
hidden_size = 1024
# 隐藏层层数
num_layers = 1
# 种类数
num_classes = 10
# 训练周期数
num_epochs = 50
# 每次取训练集中的100个数据用以训练
batch_size = 100

# 初始学习率
lr_start = 1e-3
# 最终学习率
lr_end = 1e-4
# 学习率衰减系数
lr_decay = (lr_end / lr_start)**(1. / num_epochs)
# 输入层随机置零概率
drop_in = 0.2
# 隐藏层随机置零概率
drop_hid = 0.5
# 归一化参数，
# momentum为移动平均的动量值，
momentum = 0.9
# eps为数值稳定性而加到分母上的值
eps = 1e-6

# MNIST Dataset
# 获取MNIST数据集
train_dataset = dsets.MNIST(root='../data', 
                            train=True, 
                            transform=transforms.ToTensor(),  
                            download=True)

test_dataset = dsets.MNIST(root='../data', 
                           train=False, 
                           transform=transforms.ToTensor(),
                           download=True)

# Data Loader (Input Pipeline)
# 设置训练集和测试集的dataloader，训练集打乱顺序
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# Neural Network Model
# 神经网络搭建
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1):
        super(MLP, self).__init__()
        # 获取隐藏层层数
        self.num_layers = num_layers

        # 随机将输入张量中的部分元素设置为0， p为将元素置0的概率
        self.p_in = nn.Dropout(p=drop_in)
        # 设置每层网络
        for i in range(1, self.num_layers+1):
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
        for i in range(1, self.num_layers+1):
            out = getattr(self, 'layer{}'.format(i))(out)
        # 执行输出层获取结果
        out = self.fc(out)
        return out

# 实例化神经网络
mlp = MLP(input_size, hidden_size, num_classes, num_layers=num_layers)

    
# Loss and Optimizer
# 采用交叉熵损失
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = Adam(mlp.parameters(), lr=lr_start)  


def clip_weight(parameters):
    for p in parameters:
        p = p.data
        p.clamp_(-1., 1.)


# learning rate schedule
def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = lr * lr_decay
        param_group['lr'] = lr


# Train the Model
# 开始训练
for epoch in range(num_epochs):
    # enumerate
    # 从dataloader中获取数据和标签
    for i, (images, labels) in enumerate(train_loader):  
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = mlp(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        clip_weight(mlp.parameters())
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)/batch_size, loss.item()))

    adjust_learning_rate(optimizer)

    # Test the Model
    mlp.eval()
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))
        outputs = mlp(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    mlp.train()

# Save the Trained Model
torch.save(mlp.state_dict(), 'mlp.pkl')
