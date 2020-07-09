import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import cv2

import numpy as np
import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

from resnet import resnetCIFAR10
from utils import progress_bar

# RUN : tensorboard --logdir D:\2-DOC\PROJECT\DL\ResNet\Result --host=127.0.0.1

NUM_CLASSES = 10
BATCH_SIZE = 128
PKL_NAME = "ResNet152_CIFAR10" + ".pkl"

writer = SummaryWriter('./Result')   # 数据存放在这个文件夹
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=False,transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def test():
    net = torch.load(PKL_NAME)
    
    # 测试
    with torch.no_grad():
        # 将所有的requests_grad 设置为false
        correct = 0
        total = 0

        for data in testloader:
            images, labers = data
            images, labers = images.to(device), labers.to(device)

            out = net(images)
            _, predicted = torch.max(out.data, 1)
            total += labers.size(0)
            correct += (predicted == labers).sum()
        print('Accuracy of the network on the  test images:{}%'.format(100 * correct / total)) #输出识别准确率


def train():
    LR = 0.01
    net = resnetCIFAR10(pretrained=False)
    net.train()
    # print(net)
    # 使用交叉熵作为损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器选择
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=0.0003)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=30, 
            verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    net.to(device)

    print("start Training!")
    num_epochs = 120
    correct = 0
    total = 0
    val_loss = 0
    for epoch in range(num_epochs):
        train_loss = 0
        scheduler.step(val_loss)
        writer.add_scalar('Train/Lr', optimizer.param_groups[0]['lr'], epoch)
        for batch_idx, data in enumerate(trainloader):
            inputs, labers = data
            # print (type(inputs), inputs.size())
            inputs, labers = inputs.to(device), labers.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labers)
            loss.backward()

            optimizer.step()
            # writer.add_graph(net, (inputs,))
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labers.size(0)
            correct += predicted.eq(labers).sum().item()
            niter = epoch * len(trainloader) + batch_idx
            val_loss = train_loss/(batch_idx+1)
            writer.add_scalar('Train/Loss', val_loss, niter)
            writer.add_scalar('Train/Acc', 100.*correct/total, niter)
            writer.flush()
            # 每 BATCH_SIZE 个batch显示一次当前的loss
            progress_bar(batch_idx, len(trainloader), 'Epo:%d/%d | Loss:%.3f | Acc:%.3f | Lr:%f'
                     % (epoch+1,num_epochs, train_loss/(batch_idx+1), 100.*correct/total, optimizer.param_groups[0]['lr']))
                
    print("Finished Traning")
    # 保存训练模型
    torch.save(net, PKL_NAME)

if __name__ == "__main__":
    train()
    # test()

