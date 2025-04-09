import functools
import warnings
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Sized, List, Callable, Any, Generator

import toolz
import torch
from tqdm import tqdm

from data_related import ds_operation as dso
from data_related.datasets import LazyDataSet, DataSet
from data_related.ds_operation import data_slicer
from utils import itools, tstools, ControlPanel
from utils.thread import Thread


def default_transit_fn(batch, **kwargs):
    return batch


class SelfDefinedDataSet:
    # 数据基本信息
    fea_channel = 1
    lb_channel = 1
    fea_mode = 'L'  # 读取二值图，读取图片速度将会大幅下降
    lb_mode = '1'

    # f_req_sha = (256, 256)
    # l_req_sha = (256, 256)

    def __init__(self, module: type, config: dict or ControlPanel,
                 is_train: bool = True):
        """自定义DAO数据集
        负责按照用户指定的方式读取数据集的索引以及数据本身，并提供评价指标、数据预处理方法、结果包装方法。
        读取数据集的索引体现在_get_fea_index(), _get_lb_index()方法中，在此之前需要调用_check_path()检查数据集路径。
        读取数据体现在_get_fea_reader(), _get_lb_reader()方法中。
        评价指标通过get_criterion_a()方法提供。
        数据预处理方法通过AdoptedModelName_preprocesses()提供，请将AdoptedModelName替换为本次训练使用的模型类名
        结果包装方法通过AdoptedModelName_wrap_fn()提供，请将AdoptedModelName替换为本次训练使用的模型类名
        上述所有方法均需要用户自定义。

        :param module: 实验涉及数据集类型。数据集会根据实验所用模型来自动指定数据预处理程序以及结果包装程序。
        :param config: 当前实验所属控制面板
        :param is_train: 是否处于训练模式。处于训练模式，则会加载训练和测试数据集，否则只提供测试数据集。
        """
        # 从运行动态参数中获取参数
        where = config['dataset_root']
        self.n_workers = config['n_workers']
        data_portion = config['data_portion']
        self.bulk_preprocess = config['bulk_preprocess']
        shuffle = config['shuffle']
        which = config['which_dataset']
        # 判断图片指定形状
        self.f_req_sha = tuple(config['f_req_sha'])
        self.l_req_sha = tuple(config['l_req_sha'])
        # 设置预处理程序
        if not self.bulk_preprocess:
            print(f'已选择单例预处理，将会在数据读取后立即进行预处理。本数据集将不会存储原始数据！')
            if config['lazy']:
                warnings.warn('懒加载和单例预处理冲突，优先选择单例预处理！')
                config['lazy'] = False
            self.set_preprocess(module)
            assert is_train, '测试模式下不允许单例预处理，否则将无法访问原始数据'
        else:
            self.__default_preprocesses()
        # 判断数据集懒加载程度
        self._f_lazy = self._l_lazy = config['lazy']
        # 加载训练集
        self.is_train = is_train
        # 进行数据集路径检查
        directories = self._check_path(where, which)
        indexes = ()
        if is_train:
            print(f'{self.__class__.__name__}正在获取训练索引……')
            self._train_fd, self._train_ld = directories.pop(0), directories.pop(0)
            self._train_f, self._train_l = self._get_fea_index(self._train_fd), self._get_lb_index(self._train_ld)
            # 按照数据比例切分数据集索引
            self._train_f, self._train_l = data_slicer(data_portion, shuffle, self._train_f, self._train_l)
            if self._f_lazy:
                print('对于训练集，实行懒加载特征数据集')
            else:
                indexes = (self._train_f,)
            if self._l_lazy:
                print('对于训练集，实行懒加载标签数据集')
            else:
                indexes = (*indexes, self._train_l)
                self._train_f, self._train_l = self.read_fn(
                    *indexes,
                    f_calls=self.fea_preprocesses, l_calls=self.lb_preprocesses,
                    n_workers=self.n_workers
                )
            assert len(self._train_f) == len(self._train_l), \
                f'训练集的特征集和标签集长度{len(self._train_f)}&{len(self._train_l)}不一致'
        # 加载测试集
        print(f"{self.__class__.__name__}正在获取测试索引……")
        self._test_f, self._test_l = [fn(d) for fn, d in zip([self._get_fea_index, self._get_lb_index], directories)]
        self._test_f, self._test_l = data_slicer(data_portion, shuffle, self._test_f, self._test_l)
        indexes = ()
        if self._f_lazy:
            print('对于测试集，实行懒加载特征数据集')
        else:
            indexes = (self._test_f,)
        if self._l_lazy:
            print('对于测试集，实行懒加载标签数据集')
        else:
            indexes = (*indexes, self._test_l)
            self._test_f, self._test_l = self.read_fn(
                *indexes,
                f_calls=self.fea_preprocesses, l_calls=self.lb_preprocesses,
                n_workers=self.n_workers
            )
        # 加载结果报告
        assert len(self._test_f) == len(
            self._test_l), f'测试集的特征集和标签集长度{len(self._test_f)}&{len(self._test_l)}不一致'
        print("\n************数据集读取完毕*******************")
        print(f'数据集名称为{which}')
        print(f'数据集切片大小为{data_portion}')
        if is_train:
            print(f'训练集长度为{len(self._train_f)}, 测试集长度为{len(self._test_f)}')
        else:
            print(f'测试集长度为{len(self._test_f)}')
        print("******************************************")
        # 获取特征集、标签集及其索引集的预处理程序
        if self.bulk_preprocess:
            print('已选择批量预处理，将在调用to_dataset()时进行预处理')
            self.set_preprocess(module)
        else:
            # 整理读取和预处理过后的数据，并清空预处理程序
            self._train_f = torch.vstack(self._train_f)
            self._train_l = torch.vstack(self._train_l)
            self._test_f = torch.vstack(self._test_f)
            self._test_l = torch.vstack(self._test_l)
            self.__default_preprocesses()
        # 获取结果包装程序
        self._set_wrap_fn(module)

    @abstractmethod
    def _check_path(self, root: str, which: str) -> [str, str, str, str]:
        """检查数据集路径是否正确，否则直接中断程序。

        :param root: 数据集源目录。
        :param which: 数据集批次名
        :return 返回正确的路径名。
            如果self.is_train == True，则返回[训练特征集路径，训练标签集路径，测试特征集路径，测试标签集路径]；
            否则返回[测试特征集路径，测试标签集路径]
        """
        pass

    def read_fn(
        self,
        fea_index_or_d: Iterable and Sized, lb_index_or_d: Iterable and Sized,
        f_calls: Callable = None, l_calls: Callable = None,
        n_workers: int = 1, mute: bool = False
    ):
        """加载数据集所用方法，通过调用子类自定义的Reader来获取数据读取方法，并进行数据读取。
        会根据数据集的bulk_preprocess参数来决定此时是否进行数据集预处理

        :param lb_index_or_d: 懒加载读取标签数据批所用索引
        :param fea_index_or_d: 懒加载读取特征数据批所用索引
        :param n_workers: 分配的处理机数目
        :param mute: 是否进行静默加载
        :return: 特征数据批，标签数据批
        """

        def __read_impl(reader, indexes, preprocesses, n_workers: int, mute: bool,
                        which='特征集'):
            # 如果需要进行预处理
            if preprocesses:
                wrapped_reader = toolz.compose(
                    preprocesses,
                    # functools.partial(map, reader)  # 将读取单个索引的读取器转换为针对列表的读取器
                    lambda d: [reader(d)]
                )
                # 将索引转化为压缩包内容，并升维以适应数据集级预处理程序
                # indexes = [[index] for index in indexes]
            else:
                wrapped_reader = reader
            # 进行多线程读取
            with ThreadPoolExecutor(n_workers) as pool:
                ret = pool.map(wrapped_reader, indexes)
                try:
                    if not mute:
                        return list(tqdm(
                            ret, total=len(indexes), position=0, leave=True,
                            desc=f"\r正在读取{which}……", unit="个"
                        ))
                    else:
                        return list(ret)
                except Exception as e:
                    raise e

            # if int(n_workers) > 1:
            #     # 多进程读取
            #     with multiprocessing.Pool(n_workers) as p:
            #         ret = p.map_async(reader, indexes)
            #         start_time = time.perf_counter()
            #         p.close()
            #         i = 0
            #         if not mute:
            #             while not ret.ready():
            #                 print(f'\r{which}读取中，进行了{i}秒……', end='', flush=True)
            #                 time.sleep(1)
            #                 i += 1
            #             print(f'\r{which}读取完毕，使用了{time.perf_counter() - start_time:.5f}秒', flush=True)
            #         return ret.get()
            # else:
            #     # 单进程读取
            #     if mute:
            #         return list(map(reader, indexes))
            #     else:
            #         return list(tqdm(
            #             map(reader, indexes), total=len(indexes),
            #             unit='张', desc=f'读取{which}图片中……', position=0,
            #             mininterval=1, leave=True, ncols=80
            #         ))

        # threadpool = []
        reading_args = []
        if fea_index_or_d:
            reading_args.append([
                next(self._get_fea_reader()), fea_index_or_d, f_calls,
                "特征集"
            ])
        if lb_index_or_d:
            reading_args.append([
                next(self._get_lb_reader()), lb_index_or_d, l_calls,
                "标签集"
            ])
        if n_workers < 2:
            return [__read_impl(*args[:3], 1, mute, *args[3:])
                    for args in reading_args]
        else:
            # 分两个线程处理特征集和标签集的读取
            with ThreadPoolExecutor(n_workers) as executor:
                futures = [executor.submit(
                    __read_impl,
                    *args[:3], n_workers // 2, mute, *args[3:]
                ) for args in reading_args]
            return [f.result() for f in futures]
        #     # 读取特征集
        #         if fea_index_or_d:
        #             fea_reader = next(self._get_fea_reader())
        #             ret.append(executor.submit(
        #                 __read_impl,
        #                 fea_reader, fea_index_or_d, self.fea_preprocesses,
        #                 n_workers // 2, mute, "特征集"
        #             ))
        #             # # 用户只提供单个处理机进行读取
        #             # if n_workers // 2 < 1:
        #             #     ret.append(
        #             #         __read_impl(fea_reader, fea_index_or_d, self.fea_preprocesses,
        #             #                     1, mute, "特征集")
        #             #     )
        #             # # 用户提供多个处理机进行读取
        #             # else:
        #             #     read_fea_thread = Thread(
        #             #         __read_impl,
        #             #         fea_reader, fea_index_or_d, self.fea_preprocesses,
        #             #         n_workers // 2, mute
        #             #     )
        #             #     read_fea_thread.start()
        #             #     threadpool.append(read_fea_thread)
        #         # 读取标签集
        #         if lb_index_or_d:
        #             lb_reader = next(self._get_lb_reader())
        #             ret.append(executor.submit(
        #                 __read_impl,
        #                 lb_reader, lb_index_or_d, self.lb_preprocesses,
        #                 n_workers // 2, mute, "标签集"
        #             ))
        #         # # 用户只提供单个处理机进行读取
        #         # if n_workers // 2 < 1:
        #         #     ret.append(
        #         #         __read_impl(lb_reader, lb_index_or_d, self.lb_preprocesses,
        #         #                     1, mute, "标签集")
        #         #     )
        #         # # 用户提供多个处理机进行读取
        #         # else:
        #         #     read_lb_thread = Thread(
        #         #         __read_impl,
        #         #         lb_reader, lb_index_or_d, self.lb_preprocesses,
        #         #         n_workers // 2, mute, "标签集"
        #         #     )
        #         #     read_lb_thread.start()
        #         #     threadpool.append(read_lb_thread)
        # # 确保每个线程都运行完毕，并收集他们的结果
        # # for thread in threadpool:
        # #     if thread.is_alive():
        # #         thread.join()
        # #     ret.append(thread.get_result())
        # # return ret
        # return [f.result() for f in ret]

    @abstractmethod
    def _get_fea_reader(self) -> Generator:
        """根据根目录下的特征集索引进行存储访问的数据读取器

        读取器只需要读取单个索引值。
        为了避免对数据集成包，如压缩包等，进行频繁读取关闭回收资源等操作，请通过yield的方式返回读取器。编程示例如下：

        ```
        def _get_fea_reader(self):
            # 获取.tar数据集资源
            tfile = tarfile.open(self.tarfile_name, 'r')
            # 提供数据读取器
            yield toolz.compose(*reversed([
                tfile.extractfile,
                functools.partial(read_img, mode=self.fea_mode),
            ]))
            # 进行资源回收
            tfile.close()
        ```

        :return: 数据读取器生成器
        """
        pass

    @abstractmethod
    def _get_lb_reader(self) -> Generator:
        """根据根目录下的标签集索引进行存储访问的数据读取器

        读取器只需要读取单个索引值。
        为了避免对数据集成包，如压缩包等，进行频繁读取关闭回收资源等操作，请通过yield的方式返回读取器。编程示例如下：

        ```
        def _get_lb_reader(self):
            # 获取.tar数据集资源
            tfile = tarfile.open(self.tarfile_name, 'r')
            # 提供数据读取器
            yield toolz.compose(*reversed([
                tfile.extractfile,
                functools.partial(read_img, mode=self.fea_mode),
            ]))
            # 进行资源回收
            tfile.close()
        ```

        :return: 数据读取器生成器
        """
        pass

    @abstractmethod
    def _get_fea_index(self, root) -> list:
        """读取根目录下的特征集索引

        :param root: 数据集根目录
        :return 提取出的索引列表
        """
        pass

    @abstractmethod
    def _get_lb_index(self, root) -> list:
        """读取根目录下的标签集索引

        :param root: 数据集根目录
        :return 提取出的索引列表
        """
        pass

    @staticmethod
    @abstractmethod
    def get_criterion_a() -> List[
        Callable[
            [torch.Tensor, torch.Tensor, bool],
            float or torch.Tensor
        ]
    ]:
        """获取数据集的准确率指标函数

        请注意，返回的指标函数不能有重名，否则可能会导致数据计算错误或者历史记录趋势图绘制重叠
        :return: 一系列指标函数。签名均为
        def criterion(Y_HAT, Y, size_averaged) -> float or torch.Tensor
        其中size_averaged表示是否要进行批量平均。
        """
        pass

    # def to_dataset(self, n_workers=None, collate_fn=None) -> list[LazyDataSet] or list[DataSet]:
    #     """
    #     根据自身模式，转换为合适的数据集，并对数据集进行预处理函数注册和执行。
    #     对于懒加载数据集，需要提供read_fn()，签名须为：
    #         read_fn(fea_index: Iterable[path], lb_index: Iterable[path]) -> Tuple[features: Iterable, labels: Iterable]
    #         数据加载器会自动提供数据读取路径index
    #     :return: (训练数据集、测试数据集)，两者均为pytorch框架下数据集
    #     """
    #     gen_datasets = []
    #     fl_pairs = [(self._train_f, self._train_l), (self._test_f, self._test_l)] if self.is_train \
    #         else [(self._test_f, self._test_l)]
    #     # TODO：这里的懒加载数据集生成方式不支持特征集或标签集的单独懒加载模式
    #     if self._f_lazy or self._l_lazy:
    #         # 生成懒加载数据集
    #         gen_datasets += [
    #             LazyDataSet(
    #                 f, l, i_cfn=lambda data: ([d[0] for d in data], [d[1] for d in data]),
    #                 read_fn=self.read_fn, collate_fn=collate_fn)
    #             for f, l in fl_pairs
    #         ]
    #         # 注册数据索引集的预处理程序
    #         for ds in gen_datasets:
    #             ds.register_preprocess(self.feaIndex_preprocesses, self.lbIndex_preprocesses)
    #         # if self.is_train:
    #         #     train_ds = LazyDataSet(
    #         #         self._train_f, self._train_l, read_fn=self.read_fn,
    #         #         collate_fn=index_collate_fn
    #         #     )
    #         #     train_ds.register_preprocess(
    #         #         feaIndex_calls=self.feaIndex_preprocesses, lbIndex_calls=self.lbIndex_preprocesses
    #         #     )
    #         # test_ds = LazyDataSet(
    #         #     self._test_f, self._test_l, read_fn=self.read_fn,
    #         #     collate_fn=index_collate_fn
    #         # )
    #         # test_ds.register_preprocess(
    #         #     feaIndex_calls=self.feaIndex_preprocesses, lbIndex_calls=self.lbIndex_preprocesses
    #         # )
    #         train_preprocess_desc, test_preprocess_desc = \
    #             '\r正在对训练数据索引集进行预处理……', '\r正在对测试数据索引集进行预处理……'
    #     else:
    #         # 生成数据集
    #         gen_datasets += [DataSet(f, l, collate_fn) for f, l in fl_pairs]
    #         # if self.is_train:
    #         #     train_ds = DataSet(self._train_f, self._train_l)
    #         # test_ds = DataSet(self._test_f, self._test_l)
    #         train_preprocess_desc, test_preprocess_desc = \
    #             '\r正在对训练数据集进行预处理……', '\r正在对测试数据集进行预处理……'
    #     # 注册数据集本身的预处理程序
    #     for ds in gen_datasets:
    #         ds.register_preprocess(self.fea_preprocesses, self.lb_preprocesses)
    #     # if self.is_train:
    #     #     train_ds.register_preprocess(features_calls=self.fea_preprocesses, labels_calls=self.lb_preprocesses)
    #     # test_ds.register_preprocess(features_calls=self.fea_preprocesses, labels_calls=self.lb_preprocesses)
    #     if self.bulk_preprocess:
    #         for ds, desc in zip(gen_datasets, [train_preprocess_desc, test_preprocess_desc]):
    #             print(desc, flush=True)
    #             ds.preprocess(n_workers=self.n_workers)
    #         # if self.is_train:
    #         #     train_ds.preprocess(train_preprocess_desc)
    #         # test_ds.preprocess(test_preprocess_desc)
    #     # if self.is_train:
    #     #     return train_ds, test_ds
    #     # else:
    #     #     return test_ds
    #     return gen_datasets

    def _to_dataset(self, i_cfn=None, collate_fn=None) -> list[LazyDataSet] or list[DataSet]:
        """根据自身模式，转换为合适的数据集，并对数据集进行预处理函数注册和执行。
        对于懒加载数据集，需要提供read_fn()，签名须为：
            read_fn(fea_index: Iterable[path], lb_index: Iterable[path]) -> Tuple[features: Iterable, labels: Iterable]
            数据加载器会自动提供数据读取路径index
        :return: (训练数据集、测试数据集)，两者均为pytorch框架下数据集
        """
        gen_datasets = []
        fl_pairs = [(self._train_f, self._train_l), (self._test_f, self._test_l)] if self.is_train \
            else [(self._test_f, self._test_l)]
        # TODO：这里的懒加载数据集生成方式不支持特征集或标签集的单独懒加载模式
        if self._f_lazy or self._l_lazy:
            # if i_cfn is None:
            #     i_cfn = lambda data: ([d[0] for d in data], [d[1] for d in data])
            gen_datasets += [
                LazyDataSet(
                    f, l, i_cfn=i_cfn, read_fn=self.read_fn, collate_fn=collate_fn
                ) for f, l in fl_pairs
            ]
            preprocesses = [
                self.fea_preprocesses, self.lb_preprocesses,
                self.feaIndex_preprocesses, self.lbIndex_preprocesses
            ]
            # # 注册数据索引集的预处理程序
            # for ds in gen_datasets:
            #     ds.register_preprocess(
            #         self.feaIndex_preprocesses, self.lbIndex_preprocesses,
            #         self.fea_preprocesses, self.lb_preprocesses
            #     )
            train_preprocess_desc, test_preprocess_desc = \
                '\r正在对训练数据索引集进行预处理……', '\r正在对测试数据索引集进行预处理……'
        else:
            # 生成数据集
            gen_datasets += [DataSet(f, l, collate_fn) for f, l in fl_pairs]
            train_preprocess_desc, test_preprocess_desc = \
                '\r正在对训练数据集进行预处理……', '\r正在对测试数据集进行预处理……'
            preprocesses = [self.fea_preprocesses, self.lb_preprocesses]
        # 注册数据集本身的预处理程序
        for ds in gen_datasets:
            ds.register_preprocess(*preprocesses)
        # 如进行批量预处理
        if self.bulk_preprocess:
            for ds, desc in zip(gen_datasets, [train_preprocess_desc, test_preprocess_desc]):
                print(desc, flush=True)
                ds.preprocess(n_workers=self.n_workers)
        return gen_datasets

    def to_dataloaders(self,
                       k, batch_size, i_cfn=None, collate_fn=None,
                       transit_fn: Callable = None, **dl_kwargs):
        train_ds, test_ds = self._to_dataset(i_cfn, collate_fn)
        train_portion = dl_kwargs.pop('train_portion', 0.8)
        transit_fn = transit_fn if transit_fn is not None else default_transit_fn
        # 获取数据迭代器
        test_iter = dso.to_loader(test_ds, batch_size, transit_fn, **dl_kwargs)
        if self.is_train:
            # 使用k-fold机制
            data_iter_generator = (
                [
                    dso.to_loader(
                        train_ds, batch_size, transit_fn,
                        sampler=sampler, **dl_kwargs
                    )
                    for sampler in sampler_group
                ]
                for sampler_group in dso.split_data(train_ds, k, train_portion)
            )  # 将抽取器遍历，构造加载器
            return data_iter_generator, test_iter
        return test_iter

    def _set_wrap_fn(self, module: type):
        """根据处理模型的类型，自动设置结果包装方法
        此函数会对self.wrap_fn进行赋值

        :param module: 用于处理本数据集的模型类型
        """
        if hasattr(self, f'{module.__name__}_wrap_fn'):
            self.wrap_fn = getattr(self, f'{module.__name__}_wrap_fn')
        else:
            self.wrap_fn = self.default_wrap_fn

    def set_preprocess(self, module: type):
        """根据处理模型的类型，自动指定预处理程序
        此函数会对self.module_preprocesses进行赋值，若没有编写对应模型的预处理方法，则直接赋值为空函数。

        :param module: 用于处理本数据集的模型类型
        """
        if hasattr(self, f'{module.__name__}_preprocesses'):
            exec(f'self.{module.__name__}_preprocesses()')
        else:
            self.__default_preprocesses()

    def __default_preprocesses(self) -> None:
        """默认数据集预处理程序。
        注意：此程序均为本地程序，不可被序列化（pickling）！

        类可定制化的预处理程序，需要以单个样本需要实现四个列表的填充：
        1. self.lbIndex_preprocesses：标签索引集预处理程序列表
        2. self.feaIndex_preprocesses：特征索引集预处理程序列表
        3. self.lb_preprocesses：标签集预处理程序列表
        4. self.fea_preprocesses：特征集预处理程序列表
        其中未指定本数据集为懒加载时，索引集预处理程序不会执行。
        请以数据集的角度（bulk_preprocessing）实现！

        若要定制其他网络的预处理程序，请定义函数：
        def {the_name_of_your_net}_preprocesses()
        """
        # 初始化预处理过程序
        self.lbIndex_preprocesses = None
        self.feaIndex_preprocesses = None
        self.lb_preprocesses = None
        self.fea_preprocesses = None

    def default_wrap_fn(self,
                        inputs: torch.Tensor,
                        predictions: torch.Tensor,
                        labels: torch.Tensor,
                        footnotes: list
                        ) -> Any:
        """默认结果包装方法
        将输入、预测、标签张量转换成图片，并拼接配上脚注。

        :param inputs: 输入张量
        :param predictions: 预测张量
        :param labels: 标签张量
        :param footnotes: 脚注
        :return: 包装完成的结果图
        """
        inp_s = tstools.tensor_to_img(inputs, self.fea_mode)
        pre_s = tstools.tensor_to_img(predictions, self.lb_mode)
        lb_s = tstools.tensor_to_img(labels, self.lb_mode)
        del inputs, predictions, labels
        # 制作输入、输出、标签对照图
        ret = itools.concat_imgs(
            *[
                [(inp, f'input_{inp.size}'), (pre, f'prediction_{pre.size}'),
                 (lb, f'labels_{lb.size}')]
                for inp, pre, lb in zip(inp_s, pre_s, lb_s)
            ],
            footnotes=footnotes, text_size=10, border_size=2, img_size=(192, 192),
            required_shape=(1000, 1000)
        )
        return ret

    @property
    def train_len(self):
        """数据集长度
        :return: 训练集长度，测试集长度
        """
        if self.is_train:
            return len(self._train_f)
        else:
            return 0

    @property
    def test_len(self):
        """数据集长度
        :return: 测试集长度
        """
        return len(self._test_f)
