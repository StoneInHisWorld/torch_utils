import functools
import os
from abc import abstractmethod
from typing import Iterable, Tuple, Sized, List, Callable

import torch
from tqdm import tqdm

from data_related.ds_operation import data_slicer
from data_related.datasets import LazyDataSet, DataSet
from utils.thread import Thread


class SelfDefinedDataSet:

    # 数据基本信息
    fea_channel = 1
    lb_channel = 1
    fea_mode = 'L'  # 读取二值图，读取图片速度将会大幅下降
    lb_mode = '1'
    f_required_shape = (256, 256)
    l_required_shape = (256, 256)

    def __init__(self,
                 where: str, which: str, module: type,
                 data_portion=1., shuffle=True, n_worker=1,
                 f_lazy: bool = True, l_lazy: bool = False, lazy: bool = True,
                 f_req_sha: Tuple[int, int] = (256, 256),
                 l_req_sha: Tuple[int, int] = (256, 256),
                 required_shape: Tuple[int, int] = None):
        """
        自定义DAO数据集。
        :param where: 数据集所处路径
        :param which: 实验所用数据集的文件名，用于区分同一实验，不同采样批次。
        :param module: 实验涉及数据集类型。数据集会根据实验所用模型来自动指定数据预处理程序。
        :param data_portion: 选取的数据集比例
        :param shuffle: 是否打乱数据
        :param f_lazy: 特征集懒加载参数。指定后，特征集将变为懒加载模式。
        :param l_lazy: 标签集懒加载参数。指定后，标签集将变为懒加载模式。
        :param lazy: 懒加载参数。指定后，数据集会进行懒加载，即每次通过索引取数据时，才从存储中取出数据。lazy的优先级比f/l_lazy高。
        :param f_req_sha: 需要的特征集图片形状。指定后，会将读取到的特征集图片放缩成该形状。
        :param l_req_sha: 需要的标签集图片形状。指定后，会将读取到的标签集图片放缩成该形状。
        :param required_shape: 需要的图片形状。指定后，会将读取到的图片放缩成该形状。required_shape优先级比f/l_req_sha低
        """
        # 初始化预处理过程序
        self.lbIndex_preprocesses = []
        self.feaIndex_preprocesses = []
        self.lb_preprocesses = []
        self.fea_preprocesses = []
        # 判断图片指定形状
        self.f_required_shape = required_shape
        self.l_required_shape = required_shape
        self.f_required_shape = f_req_sha
        self.l_required_shape = l_req_sha
        # 判断数据集懒加载程度
        self._f_lazy = f_lazy
        self._l_lazy = l_lazy
        self._f_lazy = lazy
        self._l_lazy = lazy
        # 进行训练数据路径检查
        self._train_fd = None
        self._train_ld = None
        self._test_fd = None
        self._test_ld = None
        self._check_path(where, which)
        # 获取特征集、标签集及其索引集的预处理程序
        self._set_preprocess(module)
        print('\n进行训练索引获取……')
        self._train_f, self._train_l = [], []
        self._get_fea_index(self._train_f, self._train_fd)
        self._get_lb_index(self._train_l, self._train_ld)
        # 按照数据比例切分数据集索引
        self._train_f, self._train_l = data_slicer(data_portion, shuffle, self._train_f, self._train_l)
        if self._f_lazy:
            print('对于训练集，实行懒加载特征数据集')
        else:
            self._train_f = self.read_fea_fn(self._train_f, n_worker)
        if self._l_lazy:
            print('对于训练集，实行懒加载标签数据集')
        # self._train_f, self._train_l = self.read_fn(self._train_f, self._train_l, n_worker)
        else:
            self._train_l = self.read_lb_fn(self._train_l, n_worker)
        # print('按照懒加载程度加载训练数据集……')
        # self._train_f = self.read_fea_fn(self._train_f, n_worker) \
        #     if not self._f_lazy else self._train_f
        # self._train_l = self.read_lb_fn(self._train_l, n_worker) \
        #     if not self._l_lazy else self._train_l
        print("\n进行测试索引获取……")
        self._test_f, self._test_l = [], []
        self._get_fea_index(self._test_f, self._test_fd)
        self._get_lb_index(self._test_l, self._test_ld)
        self._test_f, self._test_l = data_slicer(data_portion, shuffle,
                                                 self._test_f, self._test_l)
        if self._f_lazy:
            print('对于测试集，实行懒加载特征数据集')
        else:
            self._test_f = self.read_fea_fn(self._test_f, n_worker)
        if self._l_lazy:
            print('对于测试集，实行懒加载标签数据集')
        else:
            self._test_l = self.read_lb_fn(self._test_l, n_worker)
        # self._test_f, self._test_l = self.read_fn(self._test_f, self._test_l, n_worker)
        # print('按照懒加载程度加载测试数据集……')
        # self._test_f = self.read_fea_fn(self._test_f, 16) \
        #     if not self._f_lazy else self._test_f
        # self._test_l = self.read_lb_fn(self._test_l, 16) \
        #     if not self._l_lazy else self._test_l
        assert len(self._train_f) == len(self._train_l), '特征集和标签集长度须一致'
        assert len(self._test_f) == len(self._test_l), '特征集和标签集长度须一致'
        del self._train_fd, self._train_ld, self._test_fd, self._test_ld

    @abstractmethod
    def _check_path(self, root: str, which: str) -> None:
        """
        检查数据集路径是否正确，否则直接中断程序。
        本数据集要求目录结构为（请在代码中查看）：

        :param root: 数据集源目录。
        :param which: 数据集批次名
        :return: None
        """
        pass

    def read_fn(self,
                fea_index_or_d: Iterable and Sized, lb_index_or_d: Iterable and Sized,
                n_workers: int =1, mute=False
                ) -> Tuple[Iterable, Iterable]:
        """
        懒加载读取数据批所用方法。
        签名必须为：
        read_fn(fea_index: Iterable[path], lb_index: Iterable[path]) -> Tuple[features: Iterable, labels: Iterable]
        非懒加载必须使用read_fea_fn()、read_lb_fn()来读取数据。
        :param lb_index_or_d: 懒加载读取标签数据批所用索引
        :param fea_index_or_d: 懒加载读取特征数据批所用索引
        :return: 特征数据批，标签数据批
        """
        # if self._f_lazy:
        #     # 如果是懒加载，则不会开启本线程
        #     read_fea_thread = Thread(self.read_fea_fn, fea_index_or_d, n_workers // 2)
        # else:
        fea_pbar = tqdm(
            total=len(fea_index_or_d), unit='张', position=0,
            desc=f"读取特征集图片中……", mininterval=1, leave=True, ncols=80
        ) if not mute else None
        read_fea_thread = Thread(
            self.read_fea_fn, fea_index_or_d, n_workers // 2, fea_pbar
        )
        read_fea_thread.start()
        # if self._l_lazy:
        #     # 如果是懒加载，则不会开启本线程
        #     read_lb_thread = Thread(self.read_lb_fn, lb_index_or_d, n_workers // 2)
        # else:
        lb_pbar = tqdm(
            total=len(lb_index_or_d), unit='张', position=0,
            desc=f"读取标签集图片中……", mininterval=1, leave=True, ncols=80
        ) if not mute else None
        read_lb_thread = Thread(
            self.read_lb_fn, lb_index_or_d, n_workers // 2, lb_pbar
        )
        read_lb_thread.start()
        if read_fea_thread.is_alive():
            read_fea_thread.join()
        if read_lb_thread.is_alive():
            read_lb_thread.join()
        fea_index_or_d = read_fea_thread.get_result()
        lb_index_or_d = read_lb_thread.get_result()
        return fea_index_or_d, lb_index_or_d

    @staticmethod
    @abstractmethod
    def _get_fea_index(features, root) -> None:
        """
        读取根目录下的特征集索引
        :return: None
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_lb_index(labels, root) -> None:
        """
        读取根目录下的标签集索引
        :return: None
        """
        pass

    @staticmethod
    @abstractmethod
    def read_fea_fn(index: Iterable and Sized, n_worker: int = 1) -> Iterable:
        """
        加载特征集数据批所用方法
        :param n_worker: 使用的处理机数目，若>1，则开启多线程处理
        :param index: 加载特征集数据批所用索引
        :return: 读取到的特征集数据批
        """
        pass

    @staticmethod
    @abstractmethod
    def read_lb_fn(index: Iterable and Sized, n_worker: int = 1) -> Iterable:
        """
        加载标签集数据批所用方法
        :param n_worker: 使用的处理机数目，若>1，则开启多线程处理
        :param index: 加载标签集数据批所用索引
        :return: 读取到的标签集数据批
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
        :return: 一系列指标函数。签名均为criterion(Y_HAT, Y, size_averaged) -> float or torch.Tensor
                其中size_averaged表示是否要进行批量平均。
        """
        pass

    @staticmethod
    @abstractmethod
    def wrap_fn(inputs: torch.Tensor,
                predictions: torch.Tensor,
                labels: torch.Tensor,
                metric_s: torch.Tensor,
                loss_es: torch.Tensor,
                comments: List[str]
                ):
        """输出结果打包函数

        :param inputs: 预测所用到的输入批次数据
        :param predictions: 预测批次结果
        :param labels: 预测批次标签数据
        :param metric_s: 预测批次评价指标计算结果
        :param loss_es: 预测批次损失值计算结果
        :param comments: 预测批次每张结果输出图片附带注解
        :return: 打包结果
        """
        pass

    @staticmethod
    @abstractmethod
    def save_fn(result, root: str) -> None:
        """输出结果存储函数

        :param result: 输出结果图片批次
        :param root: 输出结果存储路径
        """
        pass

    def to_dataset(self) -> Tuple[LazyDataSet, LazyDataSet] or Tuple[DataSet, DataSet]:
        """
        根据自身模式，转换为合适的数据集，并对数据集进行预处理函数注册和执行。
        对于懒加载数据集，需要提供read_fn()，签名须为：
            read_fn(fea_index: Iterable[path], lb_index: Iterable[path]) -> Tuple[features: Iterable, labels: Iterable]
            数据加载器会自动提供数据读取路径index
        :return: (训练数据集、测试数据集)，两者均为pytorch框架下数据集
        """
        if self._f_lazy or self._l_lazy:
            index_collate_fn = functools.partial(
                lambda data: ([d[0] for d in data], [d[1] for d in data])
            )
            train_ds = LazyDataSet(
                self._train_f, self._train_l, read_fn=self.read_fn,
                collate_fn=index_collate_fn
            )
            train_ds.register_preprocess(
                feaIndex_calls=self.feaIndex_preprocesses, lbIndex_calls=self.lbIndex_preprocesses
            )
            test_ds = LazyDataSet(
                self._test_f, self._test_l, read_fn=self.read_fn,
                collate_fn=index_collate_fn
            )
            test_ds.register_preprocess(
                feaIndex_calls=self.feaIndex_preprocesses, lbIndex_calls=self.lbIndex_preprocesses
            )
            train_preprocess_desc, test_preprocess_desc = \
                '对训练数据索引集进行预处理……', '对测试数据索引集进行预处理……'
        else:
            train_ds = DataSet(self._train_f, self._train_l)
            test_ds = DataSet(self._test_f, self._test_l)
            train_preprocess_desc, test_preprocess_desc = \
                '对训练数据集进行预处理……', '对测试数据集进行预处理……'
        # 进行特征集本身的预处理
        train_ds.register_preprocess(features_calls=self.fea_preprocesses, labels_calls=self.lb_preprocesses)
        train_ds.preprocess(train_preprocess_desc)
        test_ds.register_preprocess(features_calls=self.fea_preprocesses, labels_calls=self.lb_preprocesses)
        test_ds.preprocess(test_preprocess_desc)
        return train_ds, test_ds

    def _set_preprocess(self, module: type):
        """
        根据处理模型的类型，自动指定预处理程序
        :param module: 用于处理本数据集的模型类型
        :return: None
        """
        if hasattr(self, f'{module.__name__}_preprocesses'):
            exec(f'self.{module.__name__}_preprocesses()')
        else:
            self.default_preprocesses()

    @abstractmethod
    def default_preprocesses(self):
        """默认数据集预处理程序。
        注意：此程序均为本地程序，不可被序列化（pickling）！

        需要实现四个列表的填充：
        1. self.lbIndex_preprocesses：标签索引集预处理程序列表
        2. self.feaIndex_preprocesses：特征索引集预处理程序列表
        3. self.lb_preprocesses：标签集预处理程序列表
        4. self.fea_preprocesses：特征集预处理程序列表
        其中索引集预处理程序若未指定本数据集为懒加载，则不会执行。

        :return: None
        """
        pass

    def __len__(self):
        return len(self._train_f)
