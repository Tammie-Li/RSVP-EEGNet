"""
makedataset.py
Created on: 2020.7.25
Author: Tammie-Li
github:
description: Complete the production of the dataset
"""
# from pykalman import KalmanFilter
import os
import numpy as np
import torch as t
import sklearn.preprocessing as pre

from scipy import signal
from scipy.io import savemat


project_path = os.getcwd()


# def Kalman1D(observations,damping=1):
#     # To return the smoothed time series data
#     observation_covariance = damping
#     initial_value_guess = observations[0]
#     transition_matrix = 1
#     transition_covariance = 0.1
#     initial_value_guess
#     kf = KalmanFilter(
#             initial_state_mean=initial_value_guess,
#             initial_state_covariance=observation_covariance,
#             observation_covariance=observation_covariance,
#             transition_covariance=transition_covariance,
#             transition_matrices=transition_matrix
#         )
#     pred_state, state_cov = kf.smooth(observations)
#     return pred_state


def make_dataset(name):
    """
    By slicing the orginal data, band-pass filtering(1~40Hz)
    down-sampling to 128Hz. Complete the production of the
    dataset.

    Args:
        name: a person who participate the experience
        save_file: the name for save data

    return:
        none
    """
    raw_path, label_path = path_operate(name)
    data, label = data_splice(raw_path, label_path)
    data_train, data_test, label_train, label_test = dataset_divide(data, label)
    # data_train, data_test = data_resample(data_train, data_test)

    file_path = project_path + '/' + name + '_1.mat'
    savemat(file_path, {'data_train': data_train,
                        'data_test': data_test,
                        'label_train': label_train,
                        'label_test': label_test,
                        })


def path_operate(name):
    """
    complete to get the absolute path

    Args:
        name: a person who participate the experience

    return:
        raw_abs_path: all absolute path of raw data
        label_abs_path: all absolute path of label
    """

    # the space to save the absolute path of raw data and label
    raw_abs_path = []
    label_abs_path = []
    raw_dir_path = str(project_path) + '/data/' + name + '/raw'
    label_dir_path = str(project_path) + '/data/' + name + '/label'

    # complete to get the absolute path
    for i in range(30):
        i = i + 1
        path_now_raw = raw_dir_path + '/' + name + '_data_' + str(i) + '.npy'
        path_now_label = label_dir_path + '/' + name + '_label_' + str(i) + '.npy'
        raw_abs_path.append(path_now_raw)
        label_abs_path.append(path_now_label)

    return raw_abs_path, label_abs_path


def data_operate(pre_data, pre_label):
    """
    complete to some operate for data, such as merge, bandpass filter and slice

    Args:
        pre_data: raw data for process
        pre_label: patch with the data

    return:
        data_end: the result of raw data
        label_end: patch with the data
    """
    # merge operate
    # (5 * 10752 * 65) -> (5 * 10240 * 65)
    list = range(10240, 10752)
    data_list = np.delete(pre_data, list, axis=1)
    # (5 * 10240 * 65) -> (51200 * 65)
    data = data_list[0, :, :]
    for i in range(4):
        add_data = data_list[i+1, :, :]
        data = np.vstack((data, add_data))
    # (51200 * 65) -> (51200 * 64)
    data = np.delete(data, 64, axis=1)
    # bandpass operate
    # a bandpass filter here
    sos = signal.butter(13, [3, 45], 'bandpass', fs=1024, output='sos')
    for i in range(64):
        data[:, i] = signal.sosfilt(sos, data[:, i])
        data[:, i] = np.flipud(data[:, i])
        data[:, i] = signal.sosfilt(sos, data[:, i])
        data[:, i] = np.flipud(data[:, i])
        # data[:, i] = Kalman1D(data[:, i], 100).flatten()
        print(i)
    # slice operate
    # (51200 * 64) -> (20 * 1024 * 64)
    k = 0;
    data_end = np.zeros((30, 1024, 64))
    label_end = np.zeros((30, 1))

    # class1:0  class2:1  notar:2
    for j in range(5):
        # class1
        data_end[k, :, :] = data[range(pre_label[j*2, 0]*205+j*10240, 1024+pre_label[j*2, 0]*205+j*10240), :]
        label_end[k, :] = 0
        # class1
        data_end[k+1, :, :] = data[range(pre_label[j*2, 1]*205+j*10240, 1024+pre_label[j*2, 1]*205+j*10240), :]
        label_end[k+1, :] = 0
        # class2
        data_end[k+2, :, :] = data[range(pre_label[j*2+1, 0]*205+j*10240, 1024+pre_label[j*2+1, 0]*205+j*10240), :]
        label_end[k+2, :] = 1
        # class2
        data_end[k+3, :, :] = data[range(pre_label[j*2+1, 1] * 205+j*10240, 1024+pre_label[j*2+1, 1]*205+j*10240), :]
        label_end[k+3, :] = 1
        # # notar
        # data_end[k+4, :, :] = data[range(2*205+j*10240, 1024+2*205+j*10240), :]
        # label_end[k+4, :] = 1
        # # notar
        # data_end[k+5, :, :] = data[range(42*205+j*10240, 1024+42*205+j*10240), :]
        # label_end[k+5, :] = 1

        k = k + 4

    return data_end, label_end


def data_splice(raw_path, label_path):
    """
    complete to splice all the data

    Args:
        raw_path: the list of raw data path
        label_path: the list of label path
    return:
        data: data for train and test
        label: label patch with data
    """

    flag = 0
    for i in range(30):
        raw_set = np.load(raw_path[i])
        label_set = np.load(label_path[i])

        if flag is 0:
            data, label = data_operate(raw_set, label_set)
            flag = flag + 1
        else:
            adddata, addlabel = data_operate(raw_set, label_set)
            data = np.vstack((data, adddata))
            label = np.vstack((label, addlabel))

    return data, label


def dataset_divide(data, label):
    """
    complete to divide the data to train dataset and test dataset,
    at the same time, normalize the data

    Args:
        data: the data which is processed
        label: label patch with data

    return:
        data_train: the data for train dataset
        data_test: the data for train dataset
        label_train: the label for test dataset
        label_test: the label for test dataset
    """
    queue = np.random.permutation(600)
    train = range(0, 440)
    test = range(440, 600)

    # add an operate()
    data_train = np.delete(data, queue[test], axis=0)
    scaler = pre.StandardScaler()
    for i in range(440):
        data_train[i, :, :] = scaler.fit_transform(data_train[i, :, :])

    data_test = np.delete(data, queue[train], axis=0)
    for i in range(160):
        data_test[i, :, :] = scaler.fit_transform(data_test[i, :, :])

    data_train = data_train.transpose(0, 2, 1)
    data_test = data_test.transpose(0, 2, 1)

    label_train = np.delete(label, queue[test], axis=0)
    label_test = np.delete(label, queue[train], axis=0)

    return data_train, data_test, label_train, label_test


def data_resample(data_train, data_test):
    """
    complete to down sampling the data to 128Hz
    Args:
        data_train: the data for train dataset
        data_test: the data for train dataset

    return:
        data_train: the data for train dataset(after resample)
        data_test: the data for train dataset(after resample)
    """
    delete_list = range(128, 1024)
    for i in range(128):
        data_train[:, :, i] = data_train[:, :, 8 * i]
    data_train = np.delete(data_train, delete_list, axis=2)
    data_train = t.from_numpy(data_train)

    for i in range(128):
        data_test[:, :, i] = data_test[:, :, 8 * i]
    data_test = np.delete(data_test, delete_list, axis=2)
    data_test = t.from_numpy(data_test)

    return data_train, data_test


make_dataset('wangshenhong')
make_dataset('wangcongcong')
make_dataset('zhangwei')
make_dataset('shepengxin')
make_dataset('hanyongkang')
make_dataset('songningning')
make_dataset('chengxin')
make_dataset('yangxin')