from models.eegnet import EEGNet
from data.dataset import TrainDataset
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import lr_scheduler

import torch
import config

import torch.nn as nn


def train_model(args, model, data_loader, criterion, optimizer, scheduler):
    """
    :param args: 训练过程参数
    :param model: 网络
    :param data_loader: 数据
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param scheduler: 学习率更新策略
    :return:
    """
    best_model_weights = model.state_dict()

    for epoch in range(args.num_epochs):
        print('---------------------Training(epoch: {%d})----------------------' % (epoch + 1))
        print('Training Epoch:%3d(%d per mini-batch)' % (epoch + 1, args.batch_size))
        model.train()
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            if args.has_cuda:
                inputs = Variable(data['inputs_train']).cuda()
                labels = Variable(data['labels_train']).cuda()
            else:
                inputs = Variable(data['inputs_train'])
                labels = Variable(data['labels_train'])
            labels = labels.view(args.batch_size)
            labels = labels.type(torch.cuda.LongTensor)
            optimizer.zero_grad()
            inputs = inputs.type(torch.cuda.FloatTensor)
            if args.has_cuda:
                output = model(inputs).cuda()
                loss = criterion(output, labels).cuda()
            else:
                output = model(inputs)
                loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % args.print_freq == args.print_freq - 1:
                print('Epoch [%d][%5d][%5d]     Loss:%8.4f       lr:%8.10f'
                      % ((epoch + 1), (i + 1), (len(data_loader)),
                         (float(running_loss / args.print_freq)), (optimizer.param_groups[0]['lr'])))
                running_loss = 0.0
        scheduler.step()
        print('\n')


if __name__ == '__main__':

    # 检测cuda状态
    HAS_CUDA = torch.cuda.is_available()
    config.args.has_cuda = HAS_CUDA
    """初始化训练参数"""
    num_train_iter = 1
    if config.args.has_cuda:
        net = EEGNet(config.args.batch_size, config.args.num_class).cuda()
    else:
        net = EEGNet(config.args.batch_size, config.args.num_class)

    optimizer = optim.Adam(net.parameters(), lr=config.args.lr)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer,
                                    step_size=config.args.step_size,
                                    gamma=config.args.gamma
                                    )
    criterion = nn.CrossEntropyLoss()

    # 加载训练集
    train_dataset = TrainDataset(path=config.args.data_path)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=config.args.batch_size,
                                   shuffle=True,
                                   num_workers=config.args.num_workers
                                   )

    # 训练网络
    train_model(config.args, net, train_data_loader, criterion, optimizer, scheduler)

    # 保存网络权重
    state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, config.args.save_weights_path)
