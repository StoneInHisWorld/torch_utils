from typing import Iterable, Callable, List

import torch
from torch.utils.data import Dataset as torch_ds, DataLoader

from utils.dataloader import LazyDataLoader


class DataSet(torch_ds):
    def __init__(self, features, labels, collate_fn: Callable = None):
        """
        普通数据集，存储数据实际内容供DataLoader进行读取。
        :param features: 数据特征集
        :param labels: 数据标签集
        :param collate_fn: 数据预处理方法。DataLoader取出数据后，使用此方法对数据进行预处理。
        """
        assert isinstance(features, Iterable) and isinstance(labels, Iterable)
        assert len(features) == len(labels), f'特征集长度{len(features)}与标签集长度{len(labels)}不等！'
        self.__features = features
        self.__labels = labels
        self.collate_fn = collate_fn

    def __getitem__(self, item):
        return self.__features[item], self.__labels[item]

    def __len__(self):
        return len(self.__features)

    def to(self, device: torch.device) -> None:
        """
        数据迁移
        :param device: 迁移目标设备
        :return:
        """
        self.__features = self.__features.to(device)
        self.__labels = self.__labels.to(device)

    def apply(self, features_calls: List[Callable[[torch.Tensor], torch.Tensor]] = None,
              labels_calls: List[Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        对数据调用某种方法，可用作数据预处理。
        :param features_calls: 需要对特征集调用的方法列表
        :param labels_calls: 需要对标签集调用的方法列表
        :return: None
        """
        if features_calls is None:
            features_calls = []
        if labels_calls is None:
            labels_calls = []
        for call in features_calls:
            self.__features = call(self.__features)
        for call in labels_calls:
            self.__labels = call(self.__labels)

    # def to_loader(self, batch_size: int = None, sampler: Iterable = None, shuffle=True,
    #               **kwargs) -> DataLoader:
    #     """
    #     根据参数生成本数据集的加载器
    #     :param sampler: 实现了__len__()的可迭代对象，用于供给下标。若不指定，则使用默认sampler.
    #     :param batch_size: 每次供给的数据量。默认为整个数据集
    #     :param shuffle: 是否打乱
    #     :param kwargs: DataLoader额外参数
    #     :return: 加载器对象
    #     """
    #     if sampler is not None:
    #         shuffle = None
    #     if not batch_size:
    #         batch_size = self.feature_shape[0]
    #     return DataLoader(
    #         self, batch_size, shuffle=shuffle, collate_fn=self.collate_fn, sampler=sampler, **kwargs
    #     )

    def get_subset(self, indices: Iterable):
        return DataSet(self[indices][0], self[indices][1])

    @property
    def feature_shape(self):
        return self.__features.shape

    @property
    def label_shape(self):
        return self.__labels.shape


class LazyDataSet(DataSet):

    def __init__(self, features, labels, load_multiple, read_fn, collate_fn=None):
        """
        懒加载数据集，只存储数据的索引供LazyDataLoader使用。
        LazyDataLoader取该数据集中实际的数据内容时，会使用`read_fn`方法进行数据内容的读取。
        :param features: 数据特征集
        :param labels: 数据标签集
        :param load_multiple: 懒加载单次加载的倍数，懒加载每次读取数据量规定为`load_multiple * batch_size`。LazyDataLoader会使用到该变量。
        :param read_fn: 数据读取方法，签名必须为：read_fn(index: Iterable[path]) -> features: Iterable。数据加载器会自动提供数据读取路径index
        :param collate_fn: 数据预处理方法
        """
        self.load_multiple = load_multiple
        self.read_fn = read_fn
        super().__init__(features, labels, collate_fn)

    # def to_loader(self, batch_size: int = None, sampler: Iterable = None, shuffle=True,
    #               **kwargs) -> LazyDataLoader:
    #     """
    #     根据参数生成本数据集的加载器
    #     :param sampler: 实现了__len__()的可迭代对象，用于供给下标。若不指定，则使用默认sampler.
    #     :param batch_size: 每次供给的数据量。默认为整个数据集
    #     :param shuffle: 是否打乱
    #     :param kwargs: DataLoader额外参数
    #     :return: 加载器对象
    #     """
    #     if sampler is not None:
    #         shuffle = None
    #     if not batch_size:
    #         batch_size = self.feature_shape[0]
    #     return LazyDataLoader(
    #         self, self.read_fn, batch_size, load_multiple=self.load_multiple, shuffle=shuffle,
    #         collate_fn=self.collate_fn, sampler=sampler, **kwargs
    #     )

