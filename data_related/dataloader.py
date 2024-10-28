import math

import dill
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader as DLoader
import utils.func.pytools as ptools

from data_related.datasets import LazyDataSet


class DataLoader(DLoader):
    """普通数据加载器"""

    def __init__(self, dataset, transit_fn, transit_kwargs,
                 batch_size=1, bkg_gen=True, max_prefetch=3,
                 **kwargs):
        """普通数据加载器

        :param dataset: 将要转化为加载器的数据集对象
        :param transit_fn: 数据批次迁移方法。
            在__iter__()方法中，每次从内存取出数据后，都会调用transit_fn对数据批次进行迁移操作。
        :param transit_kwargs: 数据供给传输函数的关键字参数。
            调用transit_fn时，一并输入到transit_fn中的关键字参数
        :param batch_size: 数据批次大小
        :param bkg_gen: 是否采用BackgroundGenerator
            BackgroundGenerator可利用多线程机制提前取出数据批
        :param max_prefetch: BackgroundGenerator提前取出的数据批数目
        :param kwargs: pytorch.utils.data.DataLoader的额外参数
        """
        self.transit_kwargs = transit_kwargs
        self.transit_fn = dill.dumps(transit_fn)
        self.bkg_gen = bkg_gen
        self.max_prefetch = max_prefetch
        super().__init__(dataset, batch_size, **kwargs)

    def __iter__(self):
        transit_fn = dill.loads(self.transit_fn)
        unwrapped_generator = (
            transit_fn(batch, **self.transit_kwargs)
            for batch in super().__iter__()
        )
        if self.bkg_gen:
            return BackgroundGenerator(unwrapped_generator, self.max_prefetch)
        else:
            return unwrapped_generator


class LazyDataLoader:

    def __init__(self, index_dataset: LazyDataSet, transit_fn,
                 batch_size=1, shuffle=True, sampler=None, transit_kwargs=None,
                 bkgGen=True, max_prefetch=3,
                 **kwargs):
        """
        数据懒加载器，适配懒加载数据集的数据加载器。每次供给时，调用索引数据集内含的读取数据方法，加载并供给索引对应的数据内容。
        :param index_dataset: 懒加载数据集
        :param batch_size: 批量大小
        :param max_load: 每次加载最大加载数量
        :param shuffle: 是否进行数据打乱
        :param sampler: 数据抽取器
        :param kwargs: DataLoader()关键字参数
        """
        # 设置DataLoader基础参数
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        # 注：不可使用dataset的collate方法，因其为索引集的校验方法
        self.__collate_fn = kwargs.pop('collate_fn', lambda d: d)
        self.__sampler = sampler
        self.__len = math.ceil(len(sampler) / batch_size) if sampler \
            else math.ceil(len(index_dataset) / batch_size)
        self.bkg_gen = bkgGen
        self.max_prefetch = max_prefetch
        self.__kwargs = kwargs
        # 设置数据转换函数
        self.__transit_fn = transit_fn
        self.transit_kwargs = transit_kwargs if transit_kwargs else {}
        # 设置数据读取函数
        self.__read_fn = index_dataset.read_fn
        # 进行索引集预处理
        self.__fea_preprocesses = index_dataset.fea_preprocesses
        self.__lb_preprocesses = index_dataset.lb_preprocesses
        # index_collate_fn = functools.partial(
        #     lambda data: ([d[0] for d in data], [d[1] for d in data])
        # )
        self.__index_loader = index_dataset.to_loader(
            batch_size, sampler, shuffle, mute=True,
        )

    def __fea_preprocess(self, fea_s):
        for call in self.__fea_preprocesses:
            fea_s = call(fea_s)
        return fea_s

    def __lb_preprocess(self, lb_s):
        for call in self.__lb_preprocesses:
            lb_s = call(lb_s)
        return lb_s

    def __iter__(self):
        # 进行预处理
        # 此处不可以使用用户提供的sampler，会发生越界问题
        # 此处数据集已经进行了打乱，因此shuffle操作也是不必要的
        def read_fn():
            for fea_indexes, lb_indexes in self.__index_loader:
                # fea_indexes, lb_indexes = batch
                fea_s, lb_s = self.__read_fn(
                    fea_indexes, lb_indexes,
                    n_workers=self.__kwargs.pop('num_workers', 2), mute=True
                )
                # 对读取到的数据进行预处理
                fea_s, lb_s = ptools.multi_process(
                    2, True, '',
                    (self.__fea_preprocess, (fea_s,), {}),
                    (self.__lb_preprocess, (lb_s,), {})
                )
                # # 对读取到的数据进行预处理
                # for call in self.__fea_preprocesses:
                #     fea_s = call(fea_s)
                # for call in self.__lb_preprocesses:
                #     lb_s = call(lb_s)
                yield self.__collate_fn((fea_s, lb_s))

        unwrapped_generator = (
            self.__transit_fn(batch, **self.transit_kwargs)
            for batch in read_fn()
        )
        # 判断是否使用BackgroundGenerator加速
        if self.bkg_gen:
            return BackgroundGenerator(unwrapped_generator, self.max_prefetch)
        else:
            return unwrapped_generator
        # for fea_indexes, lb_indexes in self.__index_loader:
        #     # TODO：试图用多线程改进效率
        #     fea_s, lb_s = self.__read_fn(fea_indexes, lb_indexes)
        #     raw_ds = DataSet(fea_s, lb_s)
        #     # 进行预处理
        #     # 此处不可以使用用户提供的sampler，会发生越界问题
        #     # 此处数据集已经进行了打乱，因此shuffle操作也是不必要的
        #     raw_ds.register_preprocess(self.__fea_preprocesses, self.__lb_preprocesses)
        #     batch_loader = raw_ds.to_loader(self.__batch_size, shuffle=False, **self.__kwargs)
        #     for X, y in batch_loader:
        #         yield X, y

    def __len__(self):
        """懒加载数据集的总长度
        计算公式为传入数据集长度整除批量大小
        :return: 长度值
        """
        return self.__len

    # def register_preprocess(self, features_calls=None, labels_calls=None):
    #     """
    #     注册预处理方法，用于数据加载器对数据进行预处理
    #     :param features_calls: 需要对特征集调用的方法列表
    #     :param labels_calls: 需要对标签集调用的方法列表
    #     """
    #     if features_calls is None:
    #         features_calls = []
    #     if labels_calls is None:
    #         labels_calls = []
    #     self.__fea_preprocesses += features_calls
    #     self.__lb_preprocesses += labels_calls
