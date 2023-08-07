#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
"""
@author:  Zj Meng
@file:    train.py.py
@time:    2023-08-07 12:19
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

# 创建网络模型
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            # nn.Flatten(),
            Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

if torch.cuda.is_available():
    print("----cuda is available!----")
else:
    print("we are using CPU!")

tudui = Tudui()
if torch.cuda.is_available():
    tudui = tudui.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
# writer = SummaryWriter("../logs_train")

# 初始化存储训练信息的列表
train_losses = []
train_times = []
test_losses = []
test_accuracies = []
test_times = []

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))
    start_time = time.time()  # 记录训练开始时间

    # 训练步骤开始
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            # writer.add_scalar("train_loss", loss.item(), total_train_step)

    end_time = time.time()  # 记录训练结束时间
    train_time = end_time - start_time  # 计算训练耗时
    train_losses.append(loss.item())
    train_times.append(train_time)


    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    test_losses.append(total_test_loss)
    test_accuracies.append(total_accuracy / test_data_size)
    test_times.append(time.time() - end_time)  # 记录测试耗时

    print("整体测试集上的Loss: {}".format(total_test_loss))
    # print("整体测试集上的正确率: {}".format(total_accuracy/test_data_size))
    # writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step = total_test_step + 1

    # torch.save(tudui, "FCTtrain_{}.pth".format(i))
    print("模型已保存")

# 绘制并保存训练 loss 曲线
plt.figure()
plt.plot(range(1, epoch+1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('train_lossCNN.png')  # 保存为图像文件
plt.show()

# 绘制并保存训练耗时曲线
plt.figure()
plt.plot(range(1, epoch+1), train_times, label='Train Time')
plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.title('Training Time')
plt.legend()
plt.savefig('train_time_torchFFT.png')  # 保存为图像文件
plt.show()

# # 绘制并保存测试耗时曲线
# plt.figure()
# plt.plot(range(1, epoch+1), test_times, label='Test Time')
# plt.xlabel('Epoch')
# plt.ylabel('Time (seconds)')
# plt.title('Test Time')
# plt.legend()
# plt.savefig('test_time.png')  # 保存为图像文件
# plt.show()

# writer.close()