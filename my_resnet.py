import os

import numpy as np
import torch
import torchvision.models.resnet as rn
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RadarDataset(Dataset):
    def __init__(self, ahr_dir, transform=None):
        self.ahr_dir = ahr_dir
        self.ahr_file = os.listdir(ahr_dir)

        self.transform = transform

    def __len__(self):
        return len(self.ahr_file)

    def __getitem__(self, idx):
        temp = np.load(self.ahr_dir + self.ahr_file[idx], allow_pickle=True).tolist()

        phase = torch.tensor(temp['rp'], dtype=torch.float32)
        label = torch.tensor(temp['hr'], dtype=torch.float32)

        if self.transform:
            # phase += math.pi
            # phase = phase/(2*math.pi)
            phase = torch.reshape(phase, [1, 24, 3])
            label = torch.reshape(label, [1])

        return phase, label


# 小修小改 resnet50
def my_resnet(net_path=''):
    temp_resnet = rn.ResNet(layers=[3, 4, 6, 3], num_classes=1, block=rn.Bottleneck)
    temp_resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    temp_resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if net_path:
        temp_resnet.load_state_dict(torch.load(net_path, map_location='cpu'))

    return temp_resnet


def precision_rate(model_path, test_path, watch_delta=233):
    # 小修小改 resnet50
    precision_resnet = my_resnet(model_path)
    precision_resnet = precision_resnet.to(device)
    precision_resnet = precision_resnet.eval()

    test_set = RadarDataset(test_path, True)  # 数据集
    batch_num = len(os.listdir(test_path))
    test_dataloader = DataLoader(test_set, batch_size=batch_num, shuffle=False)
    precision_loss = nn.L1Loss()

    for x, y in test_dataloader:
        inputs = x.to(device)
        targets = y.to(device)

        out_test = precision_resnet(inputs)
        print(precision_loss(out_test, targets))

        out_test = out_test.cpu()
        out_test = out_test.detach().numpy()

        targets = targets.cpu()
        targets = targets.detach().numpy()

        print(out_test.shape, targets.shape)

        temp = 0
        for z in range(len(test_dataloader)):
            temp += abs(out_test[z, :] - targets[z, :]) / targets[z, :]

        precision = temp / (len(test_dataloader))
        print('precision rate:', 1 - precision)

        print('out:', out_test[watch_delta:watch_delta + 10])
        print('target:', targets[watch_delta:watch_delta + 10])

        return 1 - precision


def train_resnet(data_loader):
    loss_times = []

    for x, y in data_loader:
        inputs = x.to(device)
        targets = y.to(device)

        # count_loss = 0

        for times in range(666):
            outputs = resnet(inputs)  # 前向传播
            result_loss = loss(outputs, targets)  # 输出与目标做差
            opt.zero_grad()  # 梯度归零
            result_loss.backward()  # 反向传播
            opt.step()  # 梯度下降

            print(result_loss)
            loss_times.append(result_loss.item())
            # if abs(count_loss - result_loss.item()) <= 0.001 and result_loss.item() <= 0.3:
            if result_loss.item() <= loss_size:
                break

            # count_loss = result_loss.item()

        torch.save(obj=resnet.state_dict(), f=net_save_path)
        print('resnet_models has save')

    plt.plot(loss_times)
    plt.show()

    precision = precision_rate(net_save_path, data_path + 'test/')
    print('precision rate:' + str(precision))


# def test():
#     # out_t = resnet(torch.randn(66, 1, 24, 3))
#
#     ph, la = dataSet.__getitem__(0)
#     print()
#
#     print(ph.detach().numpy()[0, 0, 0], la.shape)
#     # print(out_t.size())


data_path = '../kinds_dataset/3_24_for_test/'
net_load_path = '../resnet_models/resnet_para.pth'
net_save_path = '../resnet_models/3_24_for_test_2.pth'

if __name__ == '__main__':
    dataSet = RadarDataset(data_path + 'train/', True)  # 数据集
    train_dataloader = DataLoader(dataSet, batch_size=9337, shuffle=True)

    resnet = my_resnet(net_load_path)
    # resnet = my_resnet()
    resnet = resnet.to(device)
    resnet = resnet.train()

    loss_size = 0.0001
    precision_size = 0.83
    opt = torch.optim.SGD(resnet.parameters(), lr=0.0002, momentum=0.2)  # 优化器
    loss = nn.MSELoss()  # 损失函数

    train_resnet(train_dataloader)
