from utils.data_related import to_loader
from utils.datasets import DataSet


class LazyDataLoader:

    def __init__(self, index_dataset: DataSet, read_fn, batch_size: int = None, load_multiple: int = 1, shuffle=True,
                 features_preprocesses=None, labels_preprocesses=None, collate_fn=None, sampler=None,
                 **kwargs):
        """
        数据懒加载器。对懒加载数据集进行读取，每次供给懒加载数据集中索引对应的数据内容。
        :param index_dataset: 懒加载数据集
        :param read_fn: 数据读取方法，签名必须为：read_fn(index: Iterable[path]) -> features: Iterable。数据加载器会自动提供数据读取路径index
        :param batch_size: 批量大小
        :param load_multiple: 每次加载数量
        :param shuffle: 是否进行数据打乱
        :param collate_fn: 数据校对方法
        :param sampler: 数据抽取器
        :param kwargs: DataLoader()关键字参数
        """
        self.__batch_size = batch_size
        self.__multiple = load_multiple
        self.__shuffle = shuffle
        self.__collate_fn = collate_fn
        self.__read_fn = read_fn
        self.__sampler = sampler
        self.__kwargs = kwargs

        self.__features_preprocesses = [] if features_preprocesses is None else features_preprocesses
        self.__labels_preprocess = [] if labels_preprocesses is None else labels_preprocesses
        self.__index_loader = to_loader(index_dataset, batch_size * load_multiple, shuffle=shuffle)
        # self.__index_loader = index_dataset.to_loader(batch_size * load_multiple, shuffle=shuffle)
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
            # batch_loader = raw_ds.to_loader(
            #     self.__batch_size, self.__sampler, self.__shuffle, **self.__kwargs
            # )
            batch_loader = to_loader(
                raw_ds, self.__batch_size, self.__sampler, self.__shuffle, **self.__kwargs
            )
            for X, y in batch_loader:
                yield X, y

    def __len__(self):
        return len(self.__index_loader) * self.__multiple