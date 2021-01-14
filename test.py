from models.eegnet import EEGNet
from data.dataset import TestDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch
import config


def test_model(model, weights, args, data_loader):
    """
    :param model: 网络
    :param weights: 网络参数
    :param args: 参数
    :param data_loader: 数据
    :return: correct: 测试成功的组数
    """
    correct = 0

    # 加载模型参数
    model.load_state_dict(weights['net'])

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            if args.has_cuda:
                inputs = Variable(data['inputs_test']).cuda()
                labels = Variable(data['labels_test']).cuda()
            else:
                inputs = Variable(data['inputs_test'])
                labels = Variable(data['labels_test'])
            labels = labels.view(args.batch_size)
            labels = labels.type(torch.cuda.LongTensor)
            inputs = inputs.type(torch.cuda.FloatTensor)
            if args.has_cuda:
                outputs = model(inputs).cuda()
            else:
                outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            for index in range(args.batch_size):
                if predicted[index] == labels[index]:
                    correct += 1
    print("本轮测试的正确率为： ", float(correct/120))

    return correct


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

    # 加载训练集
    test_dataset = TestDataset(path=config.args.data_path)

    test_data_loader = DataLoader(test_dataset,
                                  batch_size=config.args.batch_size,
                                  shuffle=True,
                                  num_workers=config.args.num_workers
                                  )
    checkpoint = torch.load(config.args.finish_weights_path)
    test_model(net, checkpoint, config.args, test_data_loader)
