import threading
from concurrent.futures import ThreadPoolExecutor
import math

import dill
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader as DLoader
from utils.func.pytools import Thread

import data_related as dr
from data_related.datasets import LazyDataSet
# import threading
import queue


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
        # 设置DataLoader基础参数
        # self.__batch_size = batch_size
        # 注：不可使用dataset的collate方法，因其为索引集的校验方法
        self.collate_fn = kwargs.pop('collate_fn', lambda d: d)
        if self.collate_fn is None:
            self.collate_fn = lambda d: d
        # sampler = kwargs['sampler']
        # self.__len = math.ceil(len(sampler) / batch_size) if sampler \
        #     else math.ceil(len(index_ds) / batch_size)
        # self.__len = math.ceil(len(index_ds) / batch_size)
        # 多线程处理参数
        self.bkg_gen = bkgGen
        self.max_prefetch = max_prefetch
        # self.__kwargs = kwargs
        self.n_workers = kwargs.pop('num_workers', 1)
        if self.n_workers > 1:
            self.prefetch_factor = kwargs.pop('prefetch_factor', 2)
        # 设置数据转换函数
        # self.transit_fn = dill.dumps(transit_fn)
        self.transit_fn = transit_fn
        self.transit_kwargs = transit_kwargs if transit_kwargs else {}
        # 设置标签序列堆积函数
        i_cfn = i_cfn if i_cfn is not None else lambda data: ([d[0] for d in data], [d[1] for d in data])
        # 设置数据读取函数
        self.__read_fn = index_ds.read_fn
        # 进行索引集预处理
        self.__fea_preprocesses = index_ds.fea_preprocesses
        self.__lb_preprocesses = index_ds.lb_preprocesses
        # index_collate_fn = functools.partial(
        #     lambda data: ([d[0] for d in data], [d[1] for d in data])
        # )
        # self.__index_dl = DLoader(
        #     index_ds, batch_size, sampler=sampler,
        #     num_workers=self.n_workers, collate_fn=i_cfn, **kwargs
        # )
        # self.__index_ds = index_ds
        self.__index_dl = DLoader(
            index_ds, batch_size, collate_fn=i_cfn, **kwargs
        )

        # self.__loader = DataLoader(
        #     index_ds, transit_fn, transit_kwargs, batch_size,
        #     bkgGen, max_prefetch, collate_fn=self.__read_impl, **kwargs
        # )

    # def __fea_preprocess(self, fea_s):
    #     for call in self.__fea_preprocesses:
    #         fea_s = call(fea_s)
    #     return fea_s
    #
    # def __lb_preprocess(self, lb_s):
    #     for call in self.__lb_preprocesses:
    #         lb_s = call(lb_s)
    #     return lb_s

    # @staticmethod
    # def i_cfn(data):
    #     return [d[0] for d in data], [d[1] for d in data]

    def __read_impl(self, i_batch, n_workers):
        # i_batch = self.i_cfn(i_batch)
        try:
            fea_s, lb_s = self.__read_fn(*i_batch, n_workers=n_workers, mute=True)
            batch = self.__fea_preprocesses(fea_s), self.__lb_preprocesses(lb_s)
            batch = self.collate_fn(batch)
            return self.transit_fn(batch, **self.transit_kwargs)
        except Exception as e:
            raise e
        # return batch

    # # def __iter__(self):
    #     if self.n_workers > 2:
    #         max_prefetch = self.n_workers * self.prefetch_factor
    #         max_prefetch = min(self.max_prefetch, max_prefetch)
    #     else:
    #         max_prefetch = self.max_prefetch
    #     # 指定两个队列存储拿到的数据
    #     index_queue = queue.Queue(max_prefetch)
    #     # i_lock = threading.Lock()
    #     data_queue = queue.Queue(max_prefetch)
    #     # d_lock = threading.Lock()
    #     stop_event = threading.Event()

    #     def producer(n_workers):
    #         # 生产者线程，负责从索引数据集中读取数据，并放入数据队列中
    #         loader = (_ for _ in self.__index_dl)

    #         def __impl():
    #             index_queue.put(next(loader))

    #         if n_workers < 1:
    #             try:
    #                 while True:
    #                     __impl()
    #             except StopIteration:
    #                 pass
    #             except Exception as e:
    #                 raise e
    #         else:
    #             with ThreadPoolExecutor(n_workers) as executor:
    #                 futures = [executor.submit(__impl) for _ in range(n_workers)]
    #                 for future in futures:
    #                     try:
    #                         future.result()  # This will raise any exception that occurred in the thread
    #                     except StopIteration:
    #                         pass
    #                     except Exception as e:
    #                         raise e

    #         # for batch_indices in self.__index_dl:
    #         #     index_queue.put(batch_indices)
    #         # for _ in range(self.n_workers):
    #         #     index_queue.put(None)  # 停止所有线程
    #         # Signal the consumer that production is complete
    #         for _ in range(n_workers):
    #             index_queue.put(None)
    #         stop_event.set()

    #     def consumer(n_workers):
    #         # 消费者线程，负责从数据队列中取出数据，并进行处理
    #         # pass  # Removed unused variable
    #         #         batch_indices = index_queue.get()
    #         #         if batch_indices is None:
    #         #             break
    #         def __impl():
    #             while not stop_event.is_set() or not index_queue.empty():
    #                 # with i_lock:
    #                 batch_indices = index_queue.get()
    #                 if batch_indices is not None:
    #                     index_queue.task_done()
    #                 processed_batch = self.__read_impl(batch_indices, n_workers)
    #                 # with d_lock:
    #                 data_queue.put(processed_batch)
    #             # while True:
    #             #     if completed and index_queue.empty():
    #             #         break
    #             #     batch_indices = index_queue.get()
    #             #     # if batch_indices is None:
    #             #     #     break
    #             #     processed_batch = self.__read_impl(batch_indices)
    #             #     data_queue.put(processed_batch)

    #         if n_workers < 1:
    #             __impl()
    #         else:
    #             with ThreadPoolExecutor(n_workers) as executor:
    #                 futures = [executor.submit(__impl) for _ in range(n_workers)]
    #                 for future in futures:
    #                     future.result()  # This will raise any exception that occurred in the thread

    #         # for _ in range(self.n_workers):
    #         #     t = Thread(__impl)
    #         #     t.start()
    #         #     threads.append(t)
    #     producer_thread = Thread(producer, (self.n_workers - 1) // 2)
    #     producer_thread.daemon = True
    #     producer_thread.start()
    #     consumer_thread = Thread(consumer, (self.n_workers - 1) // 2)
    #     consumer_thread.daemon = True
    #     consumer_thread.start()

    #         #     t.join()
    #         # data_queue.put(None)  # 通知主线程已停止数据提供
    #     #
    #     # Thread(producer, (self.n_workers - 1) // 2).start()
    #     # Thread(consumer, (self.n_workers - 1) // 2).start()

    #     while not stop_event.is_set() or not data_queue.empty():
    #         # 等待数据队列不为空时，进行数据消费
    #         if data is not None:
    #             data_queue.task_done()
    #         data_queue.task_done()
    #         yield data

    # producer_thread.join()
    # consumer_thread.join()
    # if not data_queue.empty():
    #     batch = data_queue.get()
    #     # if batch is None:
    #     #     break
    #     yield batch
    # while True:
    #     batch = data_queue.get()
    #     if batch is None:
    #         break
    #     yield batch

    # return multithreaded_data_iterator(self.__index_dl, self.__read_impl, self.n_workers, self.max_prefetch)
    # return self.__loader
    # transit_fn = dill.loads(self.transit_fn)

    # def __read_impl(i_batch):
    #     fea_s, lb_s = self.__read_fn(*i_batch, n_workers=self.n_workers, mute=True)
    #     batch = self.__fea_preprocesses(fea_s), self.__lb_preprocesses(lb_s)
    #     batch = self.collate_fn(batch)
    #     self.transit_fn(batch, **self.transit_kwargs)

    # return DLoader(self.__index_ds, )
    # unwrapped_generator = (
    #     self.__read_impl(batch) for batch in self.__index_dl
    # )
    # if self.bkg_gen:
    #     return BackgroundGenerator(unwrapped_generator, self.max_prefetch)
    # else:
    #     return unwrapped_generator
    # # 进行预处理
    # # 此处不可以使用用户提供的sampler，会发生越界问题
    # # 此处数据集已经进行了打乱，因此shuffle操作也是不必要的
    # def read_fn():
    #     for fea_indexes, lb_indexes in self.__index_dl:
    #         # fea_indexes, l_i = batch
    #         fea_s, lb_s = self.__read_fn(
    #             fea_indexes, lb_indexes,
    #             n_workers=self.__kwargs.pop('num_workers', 2), mute=True
    #         )
    #         # 对读取到的数据进行预处理
    #         fea_s, lb_s = ptools.multithreading_pool(
    #             2, True, '',
    #             (self.__fea_preprocess, (fea_s,), {}),
    #             (self.__lb_preprocess, (lb_s,), {})
    #         )
    #         # # 对读取到的数据进行预处理
    #         # for call in self.__fea_preprocesses:
    #         #     fea_s = call(fea_s)
    #         # for call in self.__lb_preprocesses:
    #         #     lb_s = call(lb_s)
    #         yield self.__collate_fn((fea_s, lb_s))
    #
    # unwrapped_generator = (
    #     self.__transit_fn(batch, **self.transit_kwargs)
    #     for batch in read_fn()
    # )
    # # 判断是否使用BackgroundGenerator加速
    # if self.bkg_gen:
    #     return BackgroundGenerator(unwrapped_generator, self.max_prefetch)
    # else:
    #     return unwrapped_generator
    # for fea_indexes, l_i in self.__index_loader:
    #     fea_s, lb_s = self.__read_fn(fea_indexes, l_i)
    #     raw_ds = DataSet(fea_s, lb_s)
    #     # 进行预处理
    #     # 此处不可以使用用户提供的sampler，会发生越界问题
    #     # 此处数据集已经进行了打乱，因此shuffle操作也是不必要的
    #     raw_ds.register_preprocess(self.__fea_preprocesses, self.__lb_preprocesses)
    #     batch_loader = raw_ds.to_loader(self.__batch_size, shuffle=False, **self.__kwargs)
    #     for X, y in batch_loader:
    #         yield X, y

    # def __iter__(self):
    #
    #     def multithreaded_iterator():
    #         # Define the maximum number of prefetch batches
    #         max_prefetch = self.max_prefetch
    #         if self.n_workers > 2:
    #             max_prefetch = min(max_prefetch, self.n_workers * self.prefetch_factor)
    #
    #         # Queues for index batches and processed data
    #         index_queue = queue.Queue(max_prefetch)
    #         data_queue = queue.Queue(max_prefetch)
    #         stop_event = threading.Event()
    #
    #         def producer():
    #             """Producer thread to fetch batch indices from __index_dl."""
    #             try:
    #                 for batch_indices in self.__index_dl:
    #                     index_queue.put(batch_indices)
    #             except Exception as e:
    #                 print(f"Producer encountered an error: {e}")
    #             finally:
    #                 # Signal the consumer that production is complete
    #                 for _ in range(self.n_workers):
    #                     index_queue.put(None)
    #                 stop_event.set()
    #
    #         def consumer():
    #             """Consumer thread to process batch indices and fetch data."""
    #             while not stop_event.is_set() or not index_queue.empty():
    #                 try:
    #                     batch_indices = index_queue.get()
    #                     if batch_indices is None:
    #                         break
    #                     processed_batch = self.__read_impl(batch_indices, self.n_workers)
    #                     data_queue.put(processed_batch)
    #                 except Exception as e:
    #                     print(f"Consumer encountered an error: {e}")
    #                 finally:
    #                     index_queue.task_done()
    #
    #         # Start producer and consumer threads
    #         producer_thread = threading.Thread(target=producer, daemon=True)
    #         producer_thread.start()
    #
    #         consumer_threads = []
    #         for _ in range(self.n_workers - 1):
    #             t = threading.Thread(target=consumer, daemon=True)
    #             t.start()
    #             consumer_threads.append(t)
    #
    #         # Yield data from the data queue
    #         while not stop_event.is_set() or not data_queue.empty():
    #             try:
    #                 data = data_queue.get()
    #                 data_queue.task_done()
    #                 yield data
    #             except Exception as e:
    #                 print(f"Main thread encountered an error: {e}")
    #
    #         for t in consumer_threads:
    #             t.join()
    #
    #         # Ensure all tasks in the queues are marked as done
    #         index_queue.join()
    #         data_queue.join()
    #         for t in consumer_threads:
    #             t.join()
    #
    #     return multithreaded_iterator()

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
        """ 多线程处理逻辑，避免死锁 """
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

        with ThreadPoolExecutor(max_workers=bp_n_workers) as executor:
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
                yield batch_Q.get()
        # threadPool = [Thread(index_fetcher), Thread(batch_processor, bp_n_workers)]
        # for thread in threadPool:
        #     thread.daemon = True  # 设置为守护线程
        #     thread.start()
        # while True:
        #     batch = batch_Q.get()
        #     if batch is None:
        #         break
        #     yield batch

    def __len__(self):
        """懒加载数据集的总长度
        计算公式为传入数据集长度整除批量大小
        :return: 长度值
        """
        return len(self.__index_dl)

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
