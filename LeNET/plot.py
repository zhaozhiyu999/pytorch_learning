from torchvision.datasets import FashionMNIST
#transforms and Data 是处理数据集的
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt



train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True)

#比如说数据集一共有60000个图片，那个batch就是64个图片一组一组去划分所有的数据集
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0)

# 获得一个Batch的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break
batch_x = b_x.squeeze().numpy()  # 将四维张量移除第1维，并转换成Numpy数组。去除channel=1
batch_y = b_y.numpy()  # 将张量转换成Numpy数组
class_label = train_data.classes  # 训练集的标签
# print(class_label)
print("The size of batch in train data:", batch_x.shape)  # 每个mini-batch的维度是64*224*224

# 可视化一个Batch的图像
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray)
    plt.title(class_label[batch_y[ii]], size=10)
    plt.axis("off")
    plt.subplots_adjust(wspace=0.05)
plt.show()