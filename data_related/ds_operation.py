import random
from typing import Iterable, Sized

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader

import data_related.dataloader
from data_related.dataloader import LazyDataLoader
from data_related.datasets import DataSet, LazyDataSet


def k1_split_data(dataset: DataSet or LazyDataSet,
                  train=0.8, shuffle=True):
    """分割数据集为训练集、测试集、验证集
    test参数已删去，测试集应该与训练集独立
    返回各集合涉及下标DataLoader，只有比例>0的集合才会返回。
    :param shuffle: 每次提供的索引是否随机
    :param dataset: 分割数据集
    :param train: 训练集比例
    :return: （训练集下标DataLoader）或（训练集下标DataLoader，验证集下标DataLoader）
    """
    assert 0 < train <= 1.0, '训练集、验证集比例之和须为1'
    # 数据集分割
    print('\r正在进行数据集分割……', flush=True)
    data_len = len(dataset)
    train_len = int(data_len * train)
    ranger = (r for r in np.split(np.arange(data_len), (train_len,)) if len(r) > 0)
    return (
        [
            DataLoader(
                r, shuffle=shuffle,
                collate_fn=lambda d: d[0]  # 避免让数据升维。每次只抽取一个数字
            )
        ]
        for r in ranger
    )


def split_data(dataset: DataSet or LazyDataSet, hps, train=0.8, shuffle=True):
    # 提取超参数
    k = hps['k']
    # batch_size = hps['batch_size']
    if k == 1:
        return k1_split_data(dataset, train, shuffle)
    elif k > 1:
        return k_fold_split(dataset, k, shuffle)
    else:
        raise ValueError(f'不正确的k值={k}，k值应该大于0，且为整数！')


def default_transit_fn(batch, **kwargs):
    """默认数据供给传输函数，本函数不进行任何操作。
    DataLoader每次从内存取出数据后，都会调用本函数对批次进行预处理。
    数据传输函数用于进行数据批量的设备迁移等操作。

    :param batch: 需要预处理的数据批
    :param kwargs: 预处理所用关键字参数
    :return: 数据批次
    """
    return batch


def to_loader(dataset: DataSet or LazyDataSet,
              batch_size: int = 1,
              transit_fn=None, transit_kwargs=None,
              bkg_gen=True, max_prefetch=3,
              **kwargs):
    """根据数据集类型转化为数据集加载器。
    为了适配多进程处理，Dataset对象在转化为DataLoader对象前，会删除所携带的预处理方法。

    :param dataset: 将要转换成数据集加载器的数据集
    :param transit_fn: 数据供给传输函数，用于进行数据批量的设备迁移等操作。
        DataLoader每次从内存取出数据后，都会调用transit_fn对数据批次进行迁移操作。
        方法签名要求为def transit_fn(batch, **kwargs) -> batch:  。
        没有指定迁移方法时会指定默认迁移方法，不进行任何操作
    :param transit_kwargs: 数据供给传输函数的关键字参数。
        调用transit_fn时，一并输入到transit_fn中的关键字参数
    :param batch_size: 数据批次大小
    :param bkg_gen: 是否采用BackgroundGenerator
        BackgroundGenerator可利用多线程机制提前取出数据批
    :param max_prefetch: BackgroundGenerator提前取出的数据批数目
    :param kwargs: pytorch.utils.data.DataLoader的额外参数
    :return: 数据集加载器
    """
    if transit_fn is None:
        transit_fn = default_transit_fn
    if transit_kwargs is None:
        transit_kwargs = {}
    if isinstance(dataset, LazyDataSet):
        raise NotImplementedError('懒加载尚未完成编写！')
        # # 懒加载需要保存有数据集预处理方法
        # return LazyDataLoader(
        #     dataset, transit_fn, batch_size,
        #     shuffle=shuffle, sampler=sampler,  # 请注意，LazyDataSet提供的校正方法是针对索引集的，需要另外指定数据校正方法
        #     **kwargs
        # )
    elif isinstance(dataset, DataSet):
        dataset.pop_preprocesses()
        return data_related.dataloader.DataLoader(
            dataset, transit_fn, transit_kwargs, batch_size,
            max_prefetch=max_prefetch, bkg_gen=bkg_gen,
            collate_fn=dataset.collate_fn, **kwargs
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
    try:
        mean, std = [
            func(data, dim=list(range(2, len(data.shape))), keepdim=True)
            for func in [torch.mean, torch.std]
        ]
    except RuntimeError as e:
        if 'Input dtype must be either a floating point or complex dtype.' in e.args[0]:
            raise ValueError(f"标准化需要输入的数据为浮点数或者是复数，然而输入的数据类型为：{data.dtype}")
        else:
            raise e
    std += epsilon
    if len(data.shape) == 4:
        return F.normalize(data, mean, std)
    elif len(data.shape) < 3:
        return (data - mean) / std
    else:
        raise Exception(f'不支持的数据形状{data.shape}！')


def denormalize(data: torch.Tensor, mean, std) -> torch.Tensor:
    """进行数据反标准化

    :param data: 需要进行标准化的数据。
    :param mean: 反标准化所用均值
    :param std: 反标准化所用标准差
    :return: 反标准化的数据
    """
    return data * std + mean
