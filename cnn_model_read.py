from torch import nn
from modules import *
import numpy as np
import cv2

momentum = 0.9
eps = 1e-6

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            BinaryConv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16, momentum=momentum, eps=eps),
            nn.MaxPool2d(2),
            BinaryTanh())
        self.layer2 = nn.Sequential(
            BinaryConv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32, momentum=momentum, eps=eps),
            nn.MaxPool2d(2),
            BinaryTanh())
        self.fc = BinaryLinear(7 * 7 * 32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


cnn = CNN()
cnn.load_state_dict(torch.load('cnn.pkl'))

for para in cnn.parameters():
    print(para.shape)


img = cv2.imread('final_9.bmp', 0)
img_Data = np.array(img)
img_Data = img_Data.reshape((1,1,28,28))
img_Data = torch.from_numpy(img_Data)
img_Data = img_Data.float()

out = cnn(img_Data)
print(out.argmax())