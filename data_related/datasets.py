import time
from typing import Iterable, Callable

import torch
from torch.utils.data import Dataset as torch_ds


class DataSet(torch_ds):

    def __init__(self, features, labels, transformer,
                 collate_fn=None, device=None):
        """
        普通数据集，存储数据实际内容供DataLoader进行读取。
        :param features: 数据特征集
        :param labels: 数据标签集
        :param collate_fn: 数据预处理方法。DataLoader取出数据后，使用此方法对数据进行预处理。
        """
        assert isinstance(features, Iterable) and isinstance(labels, Iterable)
        assert len(features) == len(labels), f'特征集长度{len(features)}与标签集长度{len(labels)}不等！'
        self._features = features
        self._labels = labels
        self.transformer = transformer
        self.collate_fn = collate_fn
        self.device = device

    def __getitem__(self, item):
        return self._features[item], self._labels[item]

    def __len__(self):
        return len(self._features)

    def pop_preprocesses(self):
        """弹出所有的预处理程序。
        此方法用于对Dataset的序列化，将所有预处理的本地化方法转移到内存中。
        """
        try:
            del self.transformer
        except AttributeError:
            return

    def preprocess(self):
        """数据集对持有的特征集和标签集进行预处理

        :param n_workers: 预处理能够使用的处理机数目
        """
        start_time = time.perf_counter()
        self._features, self._labels = self.transformer.transform_data(self._features, self._labels)
        data_device = [self._features.device, self._labels.device]
        if self.device:
            print(f'\r正在进行数据迁移（{[d for d in data_device]}->{self.device}），'
                  f'如果不需要数据集整体迁移请将settings.json中的"ds_kwargs.device"设置为null')
            self._features, self._labels = self._features.to(self.device), self._labels.to(self.device)
        print(f'\r预处理完毕，使用了{time.perf_counter() - start_time:.5f}秒', flush=True)

    @property
    def feature_shape(self):
        return self._features.shape[1:]

    @property
    def label_shape(self):
        return self._labels.shape[1:]


class LazyDataSet(DataSet):

    def __init__(self, features, labels, i_cfn, reader,
                 transformer, collate_fn=None):
        """懒加载数据集，只存储数据的索引供LazyDataLoader使用
        LazyDataLoader取该数据集中实际的数据内容时，会使用`read_fn`方法进行数据内容的读取。
        :param features: 数据特征集
        :param labels: 数据标签集
        :param read_fn: 数据读取方法。
            签名必须为：
            read_fn(fea_index: Iterable[path], lb_index: Iterable[path]) -> Tuple[features: Iterable, labels: Iterable]
            数据加载器会自动提供数据读取路径index
        :param collate_fn: 数据验证方法，签名需为：签名为List[T] -> Any. DataLoader取出数据后，使用此方法对数据进行验证。
        """
        self.i_cfn = i_cfn
        self.reader = reader
        self.reader.mute = True
        super().__init__(features, labels, transformer, collate_fn)

    def preprocess(self):
        """数据集对持有的特征索引集和标签索引集进行预处理

        :param n_workers: 预处理能够使用的处理机数目
        """
        start_time = time.perf_counter()
        self._features, self._labels = self.transformer.transform_indices(self._features, self._labels)
        print(f'\r索引集预处理完毕，使用了{time.perf_counter() - start_time:.5f}秒', flush=True)
