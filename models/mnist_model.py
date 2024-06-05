from __future__ import absolute_import, division

import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        ## 输入x维度是(3, 32, 32)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5) ## 输出维度是(batchsize, 64, 28, 28) 28 = 32 - 5 + 1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(128 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 6)

    def forward(self, x):
        end_points = {}

        x = F.relu(self.conv1(x)) ## (batchsize, 64, 28, 28)
        x = F.max_pool2d(x, 2) ## (batchsize, 64, 14, 14)
        x = F.relu(self.conv2(x)) ##  (batchsize, 128, 10, 10)
        x = F.max_pool2d(x, 2) ##   (batchsize, 128, 5, 5)
        end_points['Feature'] = x #特征
        ## 这个网络的感受野是多少？  5*5 + 5*5 = 50 50 + 1 = 51
        x = x.contiguous().view(x.size(0), -1)#(batchsize, 128*5*5,1)
        x = F.relu(self.fc1(x)) ## (batchsize, 1024)
        x = F.relu(self.fc2(x)) ## (batchsize, 1024)
        end_points = x# 1024

        x = self.fc3(x)
        # end_points['Predictions'] = F.softmax(input=x, dim=-1)#输出概率

        return x, end_points


class Model_bias(nn.Module):
    def __init__(self):
        super(Model_bias, self).__init__()
        ## 输入x维度是(3, 32, 32)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1) ## 输出维度是多少？ (batchsize, 64, 32, 32) 为什么是32？ 32 = 32 - 1 + 1
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 6)

    def forward(self, x):
        end_points = {}

        x = F.relu(self.conv1(x)) ## (batchsize, 64, 32, 32)
        x = F.avg_pool2d(x, 2) ## (batchsize, 64, 16, 16)
        x = F.relu(self.conv2(x)) ## (batchsize, 128, 16, 16)
        x = F.avg_pool2d(x, 2) ##  (batchsize, 128, 8, 8)
        end_points['Feature'] = x #特征
        ## 计算这个网络的感受野=？ 1*1 + 1*1 = 2 2 + 1 = 3 感受野的计算公式=？ 3*3 + 3*3 = 18 18 + 1 = 19 为什么是19？ 19 = 32 - 8 + 1

        x = x.contiguous().view(x.size(0), -1)# 维度是多少？(batchsize, 128*8*8,1)
        x = F.relu(self.fc1(x)) ## (batchsize, 1024)
        x = F.relu(self.fc2(x)) ## (batchsize, 1024)
        end_points = x# 1024

        x = self.fc3(x)
        # end_points['Predictions'] = F.softmax(input=x, dim=-1)#输出概率

        return x, end_points

