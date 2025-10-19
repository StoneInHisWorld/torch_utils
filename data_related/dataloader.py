import functools
import queue
import warnings
from concurrent.futures import ThreadPoolExecutor

import torch
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader as DLoader

from data_related.datasets import LazyDataSet


def default_transit_fn(
        device, non_blocking, share_memory,
        indicated_transit_fn, batch, **kwargs
):
    """可自定义的数据供给传输函数
    DataLoader每次从内存取出数据后，都会调用本函数对批次进行预处理

    :param batch: 需要预处理的数据批
    :param kwargs: 预处理所用关键字参数
    :return: 数据批次
    """
    X, y = batch
    device = torch.device(device)
    if indicated_transit_fn:
        return indicated_transit_fn(batch, **kwargs)
    else:
        if X.device != device:
            X = X.to(device, non_blocking=non_blocking)
        if y.device != device:
            y = y.to(device, non_blocking=non_blocking)
        if share_memory:
            X, y = X.share_memory_(), y.share_memory_()
        return X, y


class DataLoader(DLoader):
    """普通数据加载器"""

    def __init__(self,
                 dataset, transit_fn, transit_kwargs,
                 bkg_gen=True, max_prefetch=3, device=torch.device('cpu'),
                 non_blocking=True, share_memory=True,  # 数据迁移参数
                 batch_size=1, **kwargs):  # 数据加载参数
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
        self.transit_fn = functools.partial(
            default_transit_fn, device, non_blocking, share_memory, transit_fn
        )
        self.bkg_gen = bkg_gen
        self.max_prefetch = max_prefetch
        super().__init__(dataset, batch_size, **kwargs)

    def __iter__(self):
        unwrapped_generator = (
            self.transit_fn(batch, **self.transit_kwargs)
            for batch in super().__iter__()
        )
        if self.bkg_gen:
            return BackgroundGenerator(unwrapped_generator, self.max_prefetch)
        else:
            return unwrapped_generator


class LazyDataLoader:

    def __init__(self, index_ds: LazyDataSet,
                 transit_fn, transit_kwargs=None, batch_size=1,
                 i_cfn=None, bkgGen=True, max_prefetch=3,
                 **kwargs):
        """数据懒加载器，适配懒加载数据集的数据加载器
        每次供给时，调用索引数据集内含的读取数据方法，加载并供给索引对应的数据内容。
        :param index_ds: 懒加载数据集
        :param batch_size: 批量大小
        :param max_load: 每次加载最大加载数量
        :param kwargs: DataLoader()关键字参数
        """
        # 多线程处理参数
        self.bkg_gen = bkgGen
        self.max_prefetch = max_prefetch
        self.n_workers = kwargs.pop('num_workers', 1)
        if self.n_workers > 1:
            self.prefetch_factor = kwargs.pop('prefetch_factor', 2)
        # 设置数据转换函数
        self.collate_fn = kwargs.pop('collate_fn', lambda d: d)
        if self.collate_fn is None:
            self.collate_fn = lambda d: d
        self.transit_fn = transit_fn
        self.transit_kwargs = transit_kwargs if transit_kwargs else {}
        # 设置标签序列堆积函数
        i_cfn = i_cfn if i_cfn is not None else lambda data: ([d[0] for d in data], [d[1] for d in data])
        # 设置数据读取器和转换器
        self.reader = index_ds.reader
        self.transformer = index_ds.transformer
        # 创建索引迭代器
        self.__index_dl = DLoader(index_ds, batch_size, collate_fn=i_cfn, **kwargs)

    def __read_impl(self, i_batch, n_workers):
        # print('fetch thread...')
        fea_s, lb_s = self.reader.fetch(*i_batch, n_workers=n_workers)
        # print('fetched!')
        batch = self.transformer.transform_data(fea_s, lb_s, n_workers=n_workers)
        # print('transformed!')
        batch = self.collate_fn(batch)
        return self.transit_fn(batch, **self.transit_kwargs)

    def __iter__(self):
        if self.n_workers > 2:
            return self._multi_thread_iter()
        else:
            return self._single_thread_iter()

    def _single_thread_iter(self):
        """ 单线程处理逻辑 """
        for indices in self.__index_dl:
            yield self.__read_impl(indices, 1)

    def _multi_thread_iter(self):
        """ 多线程处理逻辑 """
        if self.prefetch_factor:
            max_prefetch = min(self.max_prefetch, self.n_workers * self.prefetch_factor)
        else:
            max_prefetch = self.max_prefetch
        index_Q = queue.Queue(max_prefetch)
        batch_Q = queue.Queue(max_prefetch)
        bp_n_workers = self.n_workers - 2  # 保留两个线程用于索引数据集的读取和数据批次的发送

        def index_fetcher():
            """ 线程函数：从索引数据集中读取数据 """
            for indices in self.__index_dl:
                # print('index fetched!')
                index_Q.put(indices)
            # 结束信号
            for _ in range(bp_n_workers):
                index_Q.put(None)

        def batch_processor(n_workers):
            """ 线程函数：处理索引数据集中的数据 """
            while True:
                indices = index_Q.get()
                if indices is None:
                    break
                batch = self.__read_impl(indices, n_workers)
                batch_Q.put(batch)

        def check_futures(futures):
            """ 检查线程池中的任务是否被中断 """
            not_finished = False
            for future in futures:
                if future.done():
                    if future.exception() is not None:
                        raise future.exception()
                else:
                    not_finished = True
            return not_finished

        with ThreadPoolExecutor(max_workers=bp_n_workers + 1) as executor:
            futures = [executor.submit(index_fetcher)]
            # 提交索引数据集读取任务
            # 提交数据批次处理任务
            for _ in range(bp_n_workers):
                futures.append(executor.submit(
                    batch_processor,
                    self.prefetch_factor if self.prefetch_factor else 1
                ))
            while check_futures(futures) or not batch_Q.empty():
                # 检查线程池中的任务是否被中断
                try:
                    yield batch_Q.get(timeout=30)
                except queue.Empty:
                    warnings.warn("数据队列始终为空，正在等待数据加载……", RuntimeWarning)
                    print("请检查数据加载和预处理是否能够正常运行！")

    def __len__(self):
        """懒加载数据集的总长度
        计算公式为传入数据集长度整除批量大小
        :return: 长度值
        """
        return len(self.__index_dl)
