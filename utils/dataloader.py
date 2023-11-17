# from utils.data_related import to_loader
from typing import Tuple, Callable

from utils.datasets import DataSet


class LazyDataLoader:

    def __init__(self, index_dataset: DataSet, read_fn, batch_size: int = None, max_load: int = 1, shuffle=True,
                 features_preprocesses=None, labels_preprocesses=None, collate_fn=None, sampler=None,
                 **kwargs):
        """
        数据懒加载器。对懒加载数据集进行读取，每次供给懒加载数据集中索引对应的数据内容。
        :param index_dataset: 懒加载数据集
        :param read_fn: 数据读取方法，签名必须为：read_fn(index: Iterable[path]) -> features: Iterable。数据加载器会自动提供数据读取路径index
        :param batch_size: 批量大小
        :param max_load: 每次加载最大加载数量
        :param shuffle: 是否进行数据打乱
        :param collate_fn: 数据校对方法
        :param sampler: 数据抽取器
        :param kwargs: DataLoader()关键字参数
        """
        self.__batch_size = batch_size
        self.__max_load = max_load
        self.__shuffle = shuffle
        self.__collate_fn = collate_fn
        self.__read_fn = read_fn
        self.__sampler = sampler
        self.__len = len(index_dataset)
        self.__kwargs = kwargs

        self.__features_preprocesses = [] if features_preprocesses is None else features_preprocesses
        self.__labels_preprocess = [] if labels_preprocesses is None else labels_preprocesses
        # self.__index_loader = to_loader(index_dataset, batch_size * load_multiple, shuffle=shuffle)
        # self.__index_loader = index_dataset.to_loader(batch_size * max_load, shuffle=shuffle)
        self.__index_loader = index_dataset.to_loader(max_load, shuffle=shuffle)
        pass

    # def __iter__(self):
    #     for index, label in self.__index_loader:
    #         batch_loader = to_loader(
    #             DataSet(self.__read_fn(index), label),
    #             self.__batch_size, self.__sampler, self.__shuffle, **self.__kwargs
    #         )
    #         for X, y in batch_loader:
    #             yield X, y

    def __iter__(self):
        for index, label in self.__index_loader:
            raw_ds = DataSet(self.__read_fn(index), label)
            # 进行预处理
            raw_ds.apply(self.__features_preprocesses, self.__labels_preprocess)
            batch_loader = raw_ds.to_loader(
                self.__batch_size, self.__sampler, self.__shuffle, **self.__kwargs
            )
            # batch_loader = to_loader(
            #     raw_ds, self.__batch_size, self.__sampler, self.__shuffle, **self.__kwargs
            # )
            for X, y in batch_loader:
                yield X, y

    def __len__(self):
        # return len(self.__index_loader) * self.__max_load
        # return self.__max_load
        return self.__len

    def register_preprocess(self, features_calls=None, labels_calls=None):
        """
        注册预处理方法，用于数据加载器对数据进行预处理
        :param features_calls: 需要对特征集调用的方法列表
        :param labels_calls: 需要对标签集调用的方法列表
        :return: None
        """
        if features_calls is None:
            features_calls = []
        if labels_calls is None:
            labels_calls = []
        self.__features_preprocesses += features_calls
        self.__labels_preprocess += labels_calls

    # def add_feaprepro(self, *calls):
    #     for call in calls:
    #         self.__features_preprocesses.append(call)
    #
    # def add_labelprepro(self, *calls):
    #     for call in calls:
    #         self.__labels_preprocess.append(call)
