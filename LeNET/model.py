import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    #  约定俗成的初始化和super
    def __init__(self):
        super(LeNet, self).__init__()
        # 定义各个层,最好按照顺序去写
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)#灰度图，通道为1；步幅默认为1；out_channel代表六个卷积核
        self.sig = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)#28*28*6-->14*14*6
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)#no padding,14+0-5=9,9/1=9,9+1=10,so output 10*10*16
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)#10*10*16-->5*5*16

        self.flatten = nn.Flatten()#5*5*16=400个neurons
        self.f5 = nn.Linear(400, 120)
        self.f6 = nn.Linear(120, 84)
        self.f7 = nn.Linear(84, 10)#衣服识别，fashionmnist
    # 接着是前向传播
    def forward(self, x):
        x = self.c1(x) #c1(),传入x
        x = self.sig(x)
        x = self.s2(x)
        x = self.sig(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x
#主函数固定语法
if __name__ =="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model, (1, 28, 28)))#结果有-1，代表批量是动态的


