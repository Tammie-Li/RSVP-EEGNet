import numpy as np

from scipy.io import loadmat
from torch.utils.data import Dataset


def load_data(path):
    """
    :param path: 数据集路径
    :return: 训练源数据，测试源数据，训练标签，测试标签
    """
    src = loadmat(path)

    data_train = src['data_train']
    label_train = src['label_train']
    data_test = src['data_test']
    label_test = src['label_test']

    return data_train, data_test, label_train, label_test


def down_sample(input, in_f, out_f):
    """
    完成数据下采样
    :param input: 输入数据
    :param in_f: 输入信号的频率
    :param out_f: 输出信号的频率
    :return: 处理后的数据
    """
    delete_list = range(out_f, in_f)
    for i in range(out_f):
        input[:, :, i] = input[:, :, int(in_f / out_f) * i]
    input = np.delete(input, delete_list, axis=2)

    return input


def get_data(path):
    # 从.mat文件中读取数据
    data_train, data_test, label_train, label_test = load_data(path)
    # 数据下采样
    data_train = down_sample(data_train, 1024, 128)
    data_test = down_sample(data_test, 1024, 128)

    return data_train, data_test, label_train, label_test


class TrainDataset(Dataset):
    def __init__(self, path):
        data_train, data_test, label_train, label_test = get_data(path)
        self.inputs = data_train
        self.labels = label_train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]
        labels = self.labels[idx]

        return {'inputs_train': inputs,
                'labels_train': labels
                }


class TestDataset(Dataset):
    def __init__(self, path):
        data_train, data_test, label_train, label_test = get_data(path)
        self.inputs = data_test
        self.labels = label_test
        self.size = len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs_test = self.inputs[idx]
        labels_test = self.labels[idx]

        return {'inputs_test': inputs_test,
                'labels_test': labels_test
                }
