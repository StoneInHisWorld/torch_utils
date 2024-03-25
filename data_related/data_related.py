import random
import warnings
from typing import Iterable, Sized

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader

from data_related.dataloader import LazyDataLoader
from data_related.datasets import DataSet, LazyDataSet


def split_real_data(features: torch.Tensor, labels: torch.Tensor, train, test, valid=.0,
                    shuffle=True, requires_id=False):
    """
    分割数据集为训练集、测试集、验证集（可选）
    :param requires_id: 是否在数据每条样本前贴上ID
    :param shuffle: 是否打乱数据集
    :param labels: 标签集
    :param features: 特征集
    :param train: 训练集比例
    :param test: 测试集比例
    :param valid: 验证集比例
    :return: （训练特征集，训练标签集），（验证特征集，验证标签集），（测试特征集，测试标签集）
    """
    warnings.warn('分割真实数据集由于涉及对大量数据的操作，很容易发生内存溢出，因此建议不要使用本函数！'
                  '请使用split_data()以获得索引切分，节省计算量以及内存使用', DeprecationWarning)
    assert train + test + valid == 1.0, '训练集、测试集、验证集比例之和须为1'
    data_len = features.shape[0]
    train_len = int(data_len * train)
    valid_len = int(data_len * valid)
    test_len = int(data_len * test)
    # 将高维特征数据打上id
    if requires_id:
        features_ids = torch.tensor([
            np.ones((1, features.shape[2:])) * i
            for i in range(data_len)
        ])
        features = torch.cat((features_ids, features), 1)
        del features_ids
    # 数据集打乱
    if shuffle:
        index = torch.randint(0, data_len, (data_len,))
        features = features[index]
        labels = labels[index]
    # 数据集分割
    train_fea, valid_fea, test_fea = features.split((train_len, valid_len, test_len))
    train_labels, valid_labels, test_labels = labels.split((train_len, valid_len, test_len))
    return (train_fea, train_labels), (valid_fea, valid_labels), \
        (test_fea, test_labels)


def split_data(dataset: DataSet or LazyDataSet, train=0.8, test=0.2, valid=.0, shuffle=True):
    """
    分割数据集为训练集、测试集、验证集
    :param shuffle: 每次提供的索引是否随机
    :param dataset: 分割数据集
    :param train: 训练集比例
    :param test: 测试集比例
    :param valid: 验证集比例
    :return: 各集合涉及下标DataLoader、只有比例>0的集合才会返回。
    """
    assert train + test + valid == 1.0, '训练集、测试集、验证集比例之和须为1'
    # 数据集分割
    print('\r正在进行数据集分割……', flush=True)
    data_len = len(dataset)
    train_len = int(data_len * train)
    valid_len = int(data_len * valid)
    train_range, valid_range, test_range = np.split(np.arange(data_len), (train_len, train_len + valid_len))
    ret = (r for r in (train_range, valid_range, test_range) if len(r) > 0)
    return [
        DataLoader(
            r, shuffle=shuffle,
            collate_fn=lambda d: d[0]  # 避免让数据升维。每次只抽取一个数字
        )
        for r in ret
    ]


def to_loader(dataset: DataSet or LazyDataSet, batch_size: int = None, shuffle=True,
              sampler: Iterable = None, max_load: int = 10000,
              **kwargs):
    """
    根据数据集类型转化为数据集加载器
    :param max_load: 懒数据集加载器的最大加载量，当使用DataSet时，该参数无效
    :param sampler: 实现了__len__()的可迭代对象，用于供给下标。若不指定，则使用默认sampler，根据shuffle==True or False 提供乱序/顺序下标.
    :param dataset: 转化为加载器的数据集。
    :param batch_size: 每次供给的数据量。默认为整个数据集
    :param shuffle: 是否打乱
    :param kwargs: Dataloader额外参数
    :return: 加载器对象
    """
    if sampler is not None:
        shuffle = None
    if not batch_size:
        batch_size = dataset.feature_shape[0]
    if type(dataset) == LazyDataSet:
        return LazyDataLoader(
            dataset, batch_size,
            max_load=max_load, shuffle=shuffle, collate_fn=dataset.collate_fn,
            sampler=sampler,
            **kwargs
        )
    elif type(dataset) == DataSet:
        return DataLoader(
            dataset, batch_size,
            shuffle=shuffle, collate_fn=dataset.collate_fn, sampler=sampler,
            **kwargs
        )


def k_fold_split(dataset: DataSet or LazyDataSet, k: int = 10, shuffle: bool = True):
    """
    对数据集进行k折验证数据集切分。
    :param shuffle: 是否打乱数据集
    :param dataset: 源数据集
    :param k: 数据集拆分折数
    :return: DataLoader生成器，每次生成（训练集下标供给器，验证集下标生成器），生成k次
    """
    assert k > 1, f'k折验证需要k值大于1，而不是{k}'
    data_len = len(dataset)
    fold_size = len(dataset) // k
    total_ranger = np.random.randint(0, data_len, (data_len,)) if shuffle else np.arange(data_len)
    for i in range(k):
        train_range1, valid_range, train_range2 = np.split(
            total_ranger,
            (i * fold_size, min((i + 1) * fold_size, data_len))
        )
        train_range = np.concatenate((train_range1, train_range2), axis=0)
        del train_range1, train_range2
        yield [
            DataLoader(ranger, shuffle=True, collate_fn=lambda d: d[0])
            for ranger in (train_range, valid_range)
        ]


def data_slicer(data_portion=1., shuffle=True, *args: Sized):
    """
    数据集切分器
    :param data_portion: 数据集遍历的比例
    :param shuffle: 是否进行打乱
    :param args: 需要进行切分的数据集，可以有多个。
    :return: 切分出的数据集，具有相同的下标序列。注意：返回的数据集为元组！
    """
    assert 0 <= data_portion <= 1.0, '切分的数据集需为源数据集的子集！'
    # 验证每个需要切分的数据集的长度均相同
    data_len = len(args[0])
    for arg in args:
        assert len(arg) == data_len, '数据集切分器要求所有数据集长度均相同'
    args = list(zip(*args))
    if shuffle:
        random.shuffle(args)
    data_portion = int(data_portion * data_len)
    return zip(*args[: data_portion])  # 返回值总为元组


def normalize(data: torch.Tensor, epsilon=1e-5) -> torch.Tensor:
    """
    进行数据标准化。
    :param data: 需要进行标准化的数据。
    :param epsilon: 防止分母为0的无穷小量。
    :return: 标准化的数据
    """
    mean, std = [func(data, dim=list(range(2, len(data.shape))), keepdim=True)
                 for func in [torch.mean, torch.std]]
    if len(data.shape) == 4:
        return F.normalize(data, mean, std)
    elif len(data.shape) == 1:
        return (data - mean) / (std + epsilon)
    else:
        raise Exception(f'不支持的数据形状{data.shape}！')
