from data_related.datasets import DataSet, LazyDataSet


class LazyDataLoader:

    def __init__(self, index_dataset: LazyDataSet,
                 batch_size: int = None, shuffle=True,
                 max_load: int = 10000, collate_fn=None, sampler=None,
                 **kwargs):
        """
        数据懒加载器，适配懒加载数据集的数据加载器。每次供给时，调用索引数据集内含的读取数据方法，加载并供给索引对应的数据内容。
        :param index_dataset: 懒加载数据集
        :param batch_size: 批量大小
        :param max_load: 每次加载最大加载数量
        :param shuffle: 是否进行数据打乱
        :param collate_fn: 数据校对方法
        :param sampler: 数据抽取器
        :param kwargs: DataLoader()关键字参数
        """
        # 设置DataLoader基础参数
        self.__batch_size = batch_size
        self.__max_load = max_load
        self.__shuffle = shuffle
        self.__collate_fn = collate_fn
        self.__sampler = sampler
        self.__len = len(index_dataset) // batch_size
        self.__kwargs = kwargs
        # 设置数据读取函数
        self.__read_fn = index_dataset.read_fn
        # 进行预处理
        self.__fea_preprocesses = index_dataset.fea_preprocesses
        self.__lb_preprocesses = index_dataset.lb_preprocesses
        self.__index_loader = index_dataset.to_loader(max_load, sampler, shuffle)

    def __iter__(self):
        for fea_indexes, lb_indexes in self.__index_loader:
            # TODO：试图用多线程改进效率
            features, labels = self.__read_fn(fea_indexes, lb_indexes)
            raw_ds = DataSet(features, labels)
            # 进行预处理
            # 此处不可以使用用户提供的sampler，会发生越界问题
            # 此处数据集已经进行了打乱，因此shuffle操作也是不必要的
            raw_ds.register_preprocess(self.__fea_preprocesses, self.__lb_preprocesses)
            batch_loader = raw_ds.to_loader(self.__batch_size, shuffle=False, **self.__kwargs)
            for X, y in batch_loader:
                yield X, y

    def __len__(self):
        """
        返回懒加载数据集的总长度，计算公式为传入数据集长度整除批量大小
        :return:
        """
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
        self.__fea_preprocesses += features_calls
        self.__lb_preprocesses += labels_calls
