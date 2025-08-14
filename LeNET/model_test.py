import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet
import time


def test_data_process():
    test_data = FashionMNIST(root='./data',
                              train=False,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
    #最好1张一张去测试
    test_dataloader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=8)
    return test_dataloader


def test_model_process(model, test_dataloader):
    since = time.time()  # 记录开始时间
    # 设定测试所用到的设备，有GPU用GPU没有GPU用CPU
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print("Device used: ", device)
    # 讲模型放入到训练设备中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    test_num = 0

    # 只进行前向传播计算，不计算梯度，从而节省内存，加快运行速度
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将特征放入到测试设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到测试设备中
            test_data_y = test_data_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播过程，输入为测试数据集，输出为对每个样本的预测值
            output= model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)
            result = pre_lab.item()
            label = test_data_y.item()
            print("预测值：", result,"--------","GT：", label)
            # 如果预测正确，则准确度test_corrects加1
            test_corrects += torch.sum(pre_lab == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)

    # 计算测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为：", test_acc)

    # 计算训练和验证的耗时
    time_use = time.time() - since
    print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))


if __name__=="__main__":
    # 加载模型,用训练出来的最好参数，pth文件导入，然后去test看效果如何
    model = LeNet()
    model.load_state_dict(torch.load('best_model.pth'))
    # 加载测试数据
    test_dataloader = test_data_process()
    # 加载模型测试的函数
    test_model_process(model, test_dataloader)