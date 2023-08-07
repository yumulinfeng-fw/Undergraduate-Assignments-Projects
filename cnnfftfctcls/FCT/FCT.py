# -*- coding:utf-8 -*-
"""
@author:  Zj Meng
@file:    FFT.py
@time:    2023-08-07 22:31
@contact: ymlfvlk@gmail.com
@desc: "Welcome contact me if any questions"

"""
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from einops import rearrange, repeat

train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 定义 FCT 的模型结构
class FCT(nn.Module):
    def __init__(self, num_classes=10, patch_size=4, embed_dim=64, depth=6, num_heads=8, mlp_ratio=4):
        super(FCT, self).__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, embed_dim, 5, 1, 2),
            nn.MaxPool2d(2)
        )

        # Transformer 层
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * mlp_ratio,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=depth
        )

        # 分类层
        self.classifier = nn.Linear(embed_dim * patch_size * patch_size, num_classes)

    def forward(self, x):
        # 卷积层输出
        x = self.conv(x)

        # 切分图像为小块并展平
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        # Transformer 层输出
        x = self.transformer(x)

        # 重塑并连接分类层
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b (p1 p2 c h w)', p1=self.patch_size, p2=self.patch_size)
        x = self.classifier(x)

        return x


if torch.cuda.is_available():
    print("----cuda is available!----")
else:
    print("we are using CPU!")

# 使用 FCT 替换 Tudui
fct = FCT()
if torch.cuda.is_available():
    fct = fct.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

learning_rate = 1e-2
optimizer = torch.optim.SGD(fct.parameters(), lr=learning_rate)

total_train_step = 0
total_test_step = 0
epoch = 10

train_losses = []
train_times = []
test_losses = []
test_accuracies = []
test_times = []

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))
    start_time = time.time()

    # 训练步骤开始
    fct.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = fct(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))

    end_time = time.time()
    train_time = end_time - start_time
    train_losses.append(loss.item())
    train_times.append(train_time)

    # 测试步骤开始
    fct.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = fct(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    test_losses.append(total_test_loss)
    test_accuracies.append(total_accuracy / test_data_size)
    test_times.append(time.time() - end_time)  # 记录测试耗时

    print("整体测试集上的Loss: {}".format(total_test_loss))
    total_test_step = total_test_step + 1

    torch.save(fct, "FCT_train_{}.pth".format(i))
    print("模型已保存")
