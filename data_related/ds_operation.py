import random
from typing import Sized

import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader as DLoader

from data_related.dataloader import DataLoader, LazyDataLoader
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
            DLoader(
                r, shuffle=shuffle,
                collate_fn=lambda d: d[0]  # 避免让数据升维。每次只抽取一个数字
            )
        ]
        for r in ranger
    )


def split_data(dataset: DataSet or LazyDataSet, k, train=0.8, shuffle=True):
    # 提取超参数
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


def to_loader(
        dataset: DataSet or LazyDataSet, batch_size: int = 1, transit_fn=None,
        transit_kwargs=None, bkg_gen=True, max_prefetch=3, device=torch.device('cpu'),
        non_blocking=True, share_memory=True,  # 数据迁移参数
        **kwargs
):
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
    # if transit_fn is None:
    #     transit_fn = default_transit_fn
    if transit_kwargs is None:
        transit_kwargs = {}
    if isinstance(dataset, LazyDataSet):
        # 懒加载需要保存有数据集预处理方法
        return LazyDataLoader(
            dataset, transit_fn, transit_kwargs, batch_size,
            dataset.i_cfn, bkg_gen, max_prefetch,
            collate_fn=dataset.collate_fn, **kwargs
        )
    elif isinstance(dataset, DataSet):
        # 删除预处理方法，以便DataLoader的多进程加载
        dataset.pop_preprocesses()
        # if hasattr(dataset, 'transformer')
        # del dataset.transformer
        return DataLoader(
            dataset, transit_fn, transit_kwargs,
            bkg_gen, max_prefetch, device, non_blocking, share_memory,
            batch_size, collate_fn=dataset.collate_fn, **kwargs
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
    # print(f'\r正在进行{k}折数据集分割……', flush=True)
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
            DLoader(ranger, shuffle=True, collate_fn=lambda d: d[0])
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
    if 3 <= len(data.shape) <= 4:
        return F.normalize(data, mean, std)
    elif len(data.shape) < 3:
        return (data - mean) / std
    else:
        raise Exception(f'不支持的数据形状{data.shape}！只支持三维或者四维张量。')


def denormalize(data: torch.Tensor, mean=None, std=None, range=None) -> torch.Tensor:
    """进行数据反标准化

    当mean与std均有指定时进行反标准化，计算方法为：
    data * std + mean
    当mean与std均为None时进行数据区间的反归一化，计算方法为：
    ((data - d_min) * range / (d_max - d_min)) + low
    :param data: 需要进行标准化的数据
    :param mean: 反标准化所用均值
    :param std: 反标准化所用标准差
    :param range: 反归一化后，数据分布的区间。
    :return: 反标准化后的数据
    """
    # 如果指定了均值和方差
    if mean is not None and std is not None:
        return data * std + mean
    # 如果指定了数据区间但没指定均值和方差
    elif range is not None:
        low, high = range
        range = high - low
        # 获取最小值并扩充成源数据维度
        d_min = data.view(len(data), -1).min(1)[0]
        d_min = d_min.expand(*reversed(data.shape))
        d_min = d_min.permute(*torch.arange(d_min.ndim - 1, -1, -1))
        # 获取最大值并扩充成源数据维度
        d_max = data.view(len(data), -1).max(1)[0]
        d_max = d_max.expand(*reversed(data.shape))
        d_max = d_max.permute(*torch.arange(d_min.ndim - 1, -1, -1))
        return ((data - d_min) * range / (d_max - d_min)) + low
    else:
        raise ValueError("反标准化需要指定均值和方差，或者指定数据区间")
