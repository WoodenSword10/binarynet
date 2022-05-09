from torch import nn
from modules import *
import numpy as np
import cv2
import pandas as pd
np.set_printoptions(threshold = 1e6)
momentum = 0.9
eps = 1e-6

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            BinaryConv2d(1, 32, kernel_size=5, bias=False),
            # nn.BatchNorm2d(32, momentum=momentum, eps=eps),
            nn.MaxPool2d(2),
            BinaryTanh())
        self.layer2 = nn.Sequential(
            BinaryConv2d(32, 32, kernel_size=5, bias=False),
            # nn.BatchNorm2d(32, momentum=momentum, eps=eps),
            nn.MaxPool2d(2),
            BinaryTanh())
        self.fc = BinaryLinear(4*4*32, 10, bias=False)

    def forward(self, x):
        out = self.layer1(x)
        # print('第1层结果：\n', out, out.shape)
        out1 = out.data.numpy()
        out1 = out1.reshape(32, 12, 12)
        # print(out1)
        # writer = pd.ExcelWriter('layer1.xlsx')
        # for i in range(32):
        #     data = pd.DataFrame(out1[i])
        #     data.to_excel(writer, f'page_{i}', float_format='%.0f')
        # writer.save()
        # writer.close()
        out = self.layer2(out)
        # print('第2层结果：\n', out, out.shape)
        # out2 = out.data.numpy()
        # out2 = out2.reshape(32, 16)
        # data = pd.DataFrame(out2)
        # writer = pd.ExcelWriter('layer2.xlsx')
        # data.to_excel(writer, 'page_1', float_format='%.0f')
        # writer.save()
        # writer.close()
        out = out.view(out.size(0), -1)
        # print('view结果：\n', out, out.shape)
        out = self.fc(out)
        # print('最终结果：\n', out, out.shape)
        return out


cnn = CNN()
cnn.load_state_dict(torch.load('cnn_8.pkl'))
# count = 0
# for para in cnn.parameters():
#     data_numpy = para.data.numpy()
#     data_numpy = np.where(data_numpy >= 0, 1, 0)
#     print(data_numpy.shape, data_numpy.size)
#     np.save(f'cnn_w/w{count}.npy', data_numpy)
#     count += 1
#     # print(para)

# for i in range(10):
img = cv2.imread('final_2.bmp', 0)
img_Data = np.array(img)
img_Data = img_Data.reshape((1,1,28,28))
img_Data = torch.from_numpy(img_Data)
img_Data = img_Data.float()

out = cnn(img_Data)
print(out)
print(out.argmax())