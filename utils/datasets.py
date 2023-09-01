from typing import Iterable, Callable, List

import torch
from torch.utils.data import Dataset as torch_dataset, DataLoader


class DataSet(torch_dataset):
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
        self.__features = self.__features.to(device)
        self.__labels = self.__labels.to(device)

    def apply(self, features_calls: List[Callable[[torch.Tensor], torch.Tensor]] = None,
              labels_calls: List[Callable[[torch.Tensor], torch.Tensor]] = None):
        """
        对数据调用某种方法
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

    def to(self, device: torch.device) -> None:
        self.__features = self.__features.to(device)
        self.__labels = self.__labels.to(device)

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
        :param load_multiple: 懒加载单次加载的倍数。懒加载每次读取数据量规定为`load_multiple * batch_size`
        :param read_fn: 数据内容读取方法
        :param collate_fn: 数据预处理方法
        """
        self.load_multiple = load_multiple
        self.read_fn = read_fn
        super().__init__(features, labels, collate_fn)

