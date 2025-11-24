import functools
import time
from typing import Iterable, Callable

import torch
from torch.utils.data import Dataset as torch_ds


def default_transit_fn(device, non_blocking, share_memory, batch):
    """可自定义的数据供给传输函数
    DataLoader每次从内存取出数据后，都会调用本函数对批次进行预处理

    :param batch: 需要预处理的数据批
    :param kwargs: 预处理所用关键字参数
    :return: 数据批次
    """
    X, y = batch
    device = torch.device(device)
    if X.device != device:
        X = X.to(device, non_blocking=non_blocking)
    if y.device != device:
        y = y.to(device, non_blocking=non_blocking)
    if share_memory:
        X, y = X.share_memory_(), y.share_memory_()
    return X, y


class DataSet(torch_ds):

    def __init__(self, features, labels, transformer, is_train, bulk_transit,
                 transit_fn=None, non_blocking=True, share_memory=False,
                 transit_kwargs=None, device=None, collate_fn=None):
        """存储数据集的实际内容供DataLoader进行读取。

        :param features: 数据特征集
        :param labels: 数据标签集
        :param transformer: 数据预处理转换器
        :param collate_fn: 数据集整理方法。DataLoader获取一个批次采样下标，取出批次数据后，使用此方法对批次数据进行整理。
        # TODO：存储迁移方法使得整个数据集的迁移能够不随数据整理方法变化而失败
        :param device: 数据集目标设备。若指定此设备，则会在预处理后将整个数据集迁移到此设备
        :param is_train: 是否是训练数据集
        """
        if transit_kwargs is None:
            transit_kwargs = {}
        assert isinstance(features, Iterable) and isinstance(labels, Iterable)
        assert len(features) == len(labels), f'特征集长度{len(features)}与标签集长度{len(labels)}不等！'
        self._features = features
        self._labels = labels
        self.transformer = transformer
        self.collate_fn = collate_fn
        self.transit_kwargs = transit_kwargs
        if transit_fn:
            self.transit_fn = functools.partial(
                transit_fn, device, non_blocking, share_memory, **transit_kwargs
            )
        else:
            self.transit_fn = functools.partial(
                default_transit_fn, device, non_blocking, share_memory
            )
        self.device = device
        self.is_train = is_train
        self.bulk_transit = bulk_transit

    def __getitem__(self, item):
        batch = self._features[item], self._labels[item]
        if not self.bulk_transit:
            batch = self.transit_fn(batch)
        return batch

    def __len__(self):
        return len(self._features)

    #
    # def pop_preprocesses(self):
    #     """弹出所有的预处理程序。
    #     此方法用于对Dataset的序列化，将所有预处理的本地化方法转移到内存中。
    #     """
    #     try:
    #         del self.transformer
    #     except AttributeError:
    #         return

    # def preprocess(self, desc):
    #     """数据集对持有的特征集和标签集进行预处理。若指定了迁移设备，则会在预处理后进行整体迁移"""
    #     start_time = time.perf_counter()
    #     self._features, self._labels = self.transformer.transform_data(
    #         self._features, self._labels, self.is_train
    #     )
    #     if self.device:
    #         data_device = [self._features.device, self._labels.device]
    #         print(f'\r正在进行数据迁移（{[d for d in data_device]}->{self.device}），'
    #               f'如果不需要数据集整体迁移请将settings.json中的"ds_kwargs.device"设置为null', flush=True, end="")
    #         self._features, self._labels = self._features.to(self.device), self._labels.to(self.device)
    #     print(f'\r{desc}预处理完毕，使用了{time.perf_counter() - start_time:.5f}秒', flush=True)

    def preprocess(self, desc):
        """数据集对持有的特征集和标签集进行预处理。若指定了迁移设备，则会在预处理后进行整体迁移"""
        start_time = time.perf_counter()
        self._features, self._labels = self.transformer.transform_data(
            self._features, self._labels, self.is_train
        )
        if self.bulk_transit:
            self._features, self._labels = self.transit_fn((self._features, self._labels))
            print(f'\r已将数据集整体迁移至{self.device}', flush=True, end="")
        print(f'\r{desc}预处理完毕，使用了{time.perf_counter() - start_time:.5f}秒', flush=True)


class LazyDataSet(DataSet):

    def __init__(self, i_cfn, reader, *ds_args, **ds_kwargs):
        """懒加载数据集，存储数据的索引供LazyDataLoader使用。
        LazyDataLoader取该数据集中实际的数据内容时，会使用`reader`进行数据内容的读取。

        :param i_cfn: 索引集整理方法。LazyDataLoader选择采样索引批时，会调用此方法对索引批次进行整理。
        :param reader: 数据读取器。
            LazyDataLoader取出采样批次的索引后，会调用此方法进行数据的读取。签名必须为：
                def reader(fea_index: Iterable[path], lb_index: Iterable[path])
                    -> Tuple[features: Iterable, labels: Iterable]
        :param labels: 数据标签集
        :param ds_args: DataSet初始化的位置参数。
        :param ds_kwargs: DataSet初始化的关键字参数。
            device参数在本类中不会奏效，因为索引集预处理后不会进行批量迁移。
        """
        self.i_cfn = i_cfn
        self.reader = reader
        self.reader.mute = True
        super().__init__(**ds_kwargs)

    def preprocess(self, desc):
        """数据集对持有的特征索引集和标签索引集进行预处理"""
        start_time = time.perf_counter()
        self._features, self._labels = self.transformer.transform_indices(
            self._features, self._labels, is_train=self.is_train
        )
        print(f'\r{desc}预处理完毕，使用了{time.perf_counter() - start_time:.5f}秒', flush=True)
