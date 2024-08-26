import functools
from abc import abstractmethod
from multiprocessing import Pool
from typing import Iterable, Tuple, Sized, List, Callable, Any

import toolz
import torch
from tqdm import tqdm

import utils.func.img_tools as itools
import utils.func.tensor_tools as tstools
from data_related.datasets import LazyDataSet, DataSet
from data_related.ds_operation import data_slicer
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
                 where: str, which: str, module: type, control_panel,
                 shuffle=True, f_lazy: bool = True, l_lazy: bool = False,
                 f_req_sha: Tuple[int, int] = (256, 256),
                 l_req_sha: Tuple[int, int] = (256, 256),
                 required_shape: Tuple[int, int] = None):
        """
        自定义DAO数据集。
        :param where: 数据集所处路径
        :param which: 实验所用数据集的文件名，用于区分同一实验，不同采样批次。
        :param module: 实验涉及数据集类型。数据集会根据实验所用模型来自动指定数据预处理程序。
        :param shuffle: 在划分数据集前是否打乱数据
        :param f_lazy: 特征集懒加载参数。指定后，特征集将变为懒加载模式。
        :param l_lazy: 标签集懒加载参数。指定后，标签集将变为懒加载模式。
        :param f_req_sha: 需要的特征集图片形状。指定后，会将读取到的特征集图片放缩成该形状。
        :param l_req_sha: 需要的标签集图片形状。指定后，会将读取到的标签集图片放缩成该形状。
        :param required_shape: 需要的图片形状。指定后，会将读取到的图片放缩成该形状。required_shape优先级比f/l_req_sha低
        """
        # 从运行动态参数中获取参数
        n_worker = control_panel['n_workers']
        lazy = control_panel['lazy']
        data_portion = control_panel['data_portion']
        bulk_preprocess = control_panel['bulk_preprocess']
        # # 初始化预处理过程序
        # self.lbIndex_preprocesses = toolz.compose()
        # self.feaIndex_preprocesses = toolz.compose()
        # self.lb_preprocesses = toolz.compose()
        # self.fea_preprocesses = toolz.compose()
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
        print('\n进行训练索引获取……')
        self._train_f, self._train_l = [], []
        self._get_fea_index(self._train_f, self._train_fd)
        self._get_lb_index(self._train_l, self._train_ld)
        # 按照数据比例切分数据集索引
        self._train_f, self._train_l = data_slicer(data_portion, shuffle, self._train_f, self._train_l)
        # 进行数据集加载
        # if n_worker > 2:
        #     def data_read_impl(prompt, fn, indexes, preprocessing):
        #         fn = toolz.compose(preprocessing, fn)
        #         with Pool(n_worker - 1) as p:
        #             ls_ret = list(tqdm(
        #                 p.imap(fn, [[f] for f in indexes]),
        #                 total=len(indexes), unit='张', position=0,
        #                 desc=prompt, mininterval=1, leave=True, ncols=80
        #             ))
        #         return torch.vstack(ls_ret)
        # else:
        #     def data_read_impl(prompt, fn, indexes, preprocessing):
        #         fn = toolz.compose(preprocessing, fn)
        #         ls_ret = list(tqdm(
        #             map(fn, [[i] for i in indexes]),
        #             total=len(indexes), unit='张', position=0,
        #             desc=prompt, mininterval=1, leave=True, ncols=80
        #         ))
        #         return torch.vstack(ls_ret)

        # def data_read_impl(prompt, fn, indexes, preprocessing):
        #     fn = toolz.compose(preprocessing, fn)
        #     ls_ret = list(tqdm(
        #         map(fn, [[i] for i in indexes]),
        #         total=len(indexes), unit='张', position=0,
        #         desc=prompt, mininterval=1, leave=True, ncols=80
        #     ))
        #     return torch.vstack(ls_ret)

        if not bulk_preprocess:
            self._set_preprocess(module)
        else:
            self.__default_preprocesses()
        # 加载训练集
        if self._f_lazy:
            print('对于训练集，实行懒加载特征数据集')
        else:
            # train_f = self._train_f
            # del self._train_f
            # self._train_f = data_read_impl(
            #     "读取训练集的特征集图片中……", self.read_fea_fn, train_f,
            #     self.fea_preprocesses
            # )
            # pbar = tqdm(
            #     total=len(self._train_f), unit='张', position=0,
            #     desc=f"读取训练集的特征集图片中……", mininterval=1, leave=True, ncols=80
            # )
            self._train_f = self.read_fea_fn(
                self._train_f, n_worker, preprocesses=self.fea_preprocesses
            )
            # self._train_f = self.read_fea_fn(self._train_f, n_worker, pbar)
            # pbar.close()
        if self._l_lazy:
            print('对于训练集，实行懒加载标签数据集')
        else:
            # train_l = self._train_l
            # del self._train_l
            # self._train_l = data_read_impl(
            #     "读取训练集的标签集图片中……", self.read_lb_fn, train_l,
            #     self.lb_preprocesses
            # )
            # pbar = tqdm(
            #     total=len(self._train_l), unit='张', position=0,
            #     desc=f"读取训练集的标签集图片中……", mininterval=1, leave=True, ncols=80
            # )
            self._train_l = self.read_lb_fn(
                self._train_l, n_worker, preprocesses=self.lb_preprocesses
            )
            # pbar.close()
        # 加载测试集
        print("\n进行测试索引获取……")
        self._test_f, self._test_l = [], []
        self._get_fea_index(self._test_f, self._test_fd)
        self._get_lb_index(self._test_l, self._test_ld)
        self._test_f, self._test_l = data_slicer(data_portion, shuffle,
                                                 self._test_f, self._test_l)
        if self._f_lazy:
            print('对于测试集，实行懒加载特征数据集')
        else:
            # test_f = self._test_f
            # del self._test_f
            # self._test_f = data_read_impl(
            #     "读取测试集的特征集图片中……", self.read_fea_fn, test_f,
            #     self.fea_preprocesses
            # )
            # pbar = tqdm(
            #     total=len(self._test_f), unit='张', position=0,
            #     desc=f"读取测试集的特征集图片中……", mininterval=1, leave=True, ncols=80
            # )
            #
            # def update_pbar(ret):
            #     print('called back!')
            #     pbar.update(1)
            # with Pool(n_worker - 1) as p:
            #     ret = list(tqdm(
            #         p.imap(self.read_fea_fn, [[f] for f in self._test_f]),
            #         total=len(self._test_f)
            #     ))
                # p.close()
                # p.join()

            # self._test_f = ret
            self._test_f = self.read_fea_fn(
                self._test_f, n_worker, preprocesses=self.fea_preprocesses
            )
            # pbar.close()
        if self._l_lazy:
            print('对于测试集，实行懒加载标签数据集')
        else:
            self._test_l = self.read_lb_fn(
                self._test_l, n_worker, preprocesses=self.lb_preprocesses
            )
            # test_l = self._test_l
            # del self._test_l
            # self._test_l = data_read_impl(
            #     "读取测试集的标签集图片中……", self.read_lb_fn, test_l,
            #     self.lb_preprocesses
            # )
            # pbar = tqdm(
            #     total=len(self._test_l), unit='张', position=0,
            #     desc=f"读取测试集的标签集图片中……", mininterval=1, leave=True, ncols=80
            # )
            # self._test_l = self.read_lb_fn(self._test_l, n_worker, pbar)
            # pbar.close()
        assert len(self._train_f) == len(self._train_l), \
            f'训练集的特征集和标签集长度{len(self._train_f)}&{len(self._train_l)}不一致'
        assert len(self._test_f) == len(self._test_l), \
            f'测试集的特征集和标签集长度{len(self._test_f)}&{len(self._test_l)}不一致'
        print(f'训练集长度为{len(self._train_f)}')
        print(f'测试集长度为{len(self._test_f)}')
        # 获取特征集、标签集及其索引集的预处理程序
        if bulk_preprocess:
            self._set_preprocess(module)
        else:
            self._train_f = torch.vstack(self._train_f)
            self._train_l = torch.vstack(self._train_l)
            self._test_f = torch.vstack(self._test_f)
            self._test_l = torch.vstack(self._test_l)
            self.__default_preprocesses()
        # 获取结果包装程序
        self._set_wrap_fn(module)
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
                n_workers: int = 1, mute: bool = False
                ) -> Tuple[Iterable, Iterable]:
        """懒加载读取数据批所用方法。
        非懒加载必须分别使用read_fea_fn()、read_lb_fn()来读取特征集、标签集数据。

        :param mute: 是否进行静默加载
        :param n_workers: 分配的处理机数目
        :param lb_index_or_d: 懒加载读取标签数据批所用索引
        :param fea_index_or_d: 懒加载读取特征数据批所用索引
        :return: 特征数据批，标签数据批
        """
        # # 开启进度条
        # pbar = tqdm(
        #     total=len(fea_index_or_d) + len(lb_index_or_d), unit='张',
        #     position=0, desc=f"读取数据集中……", mininterval=1, leave=True, ncols=80
        # ) if not mute else None
        # pbar_Q = SimpleQueue()
        # # 创建子进程
        # pool = Pool(processes=2)  # 子进程数
        # fea_res = pool.apply_async(
        #     self.read_fea_fn, fea_index_or_d, (n_workers - 1) // 2, pbar_Q
        # )
        # lb_res = pool.apply_async(
        #     self.read_lb_fn, lb_index_or_d, (n_workers - 1) // 2, pbar_Q
        # )
        # pool.shutdown()
        # # 接受进度条信息
        # while True:
        #     item = pbar_Q.get()
        #     if item is None:
        #         if pool.isTerminated():
        #             break
        #     # elif isinstance(item, Exception):
        #     #     raise InterruptedError('数据集读取过程中某处触发了异常，请根据上条Trackback信息进行排查！')
        #     elif isinstance(item, int):
        #         pbar.update(item)
        #     else:
        #         raise ValueError(f'不识别的信号{item}')
        # fea_index_or_d = fea_res.get()
        # lb_index_or_d = lb_res.get()
        # return fea_index_or_d, lb_index_or_d
        fea_pbar = tqdm(
            total=len(fea_index_or_d), unit='张', position=0,
            desc=f"读取特征集图片中……", mininterval=1, leave=True, ncols=80
        ) if not mute else None
        read_fea_thread = Thread(
            self.read_fea_fn, fea_index_or_d, n_workers // 2, fea_pbar
        )
        read_fea_thread.start()
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

    @abstractmethod
    def read_fea_fn(self, index: Iterable and Sized,
                    n_worker: int = 1, pbar=None) -> Iterable:
        """加载特征集数据批所用方法

        :param index: 加载特征集数据批所用索引
        :param n_worker: 使用的处理机数目，若>1，则开启多线程处理
        :param pbar: 加载特征集时所用的进度条
        :return: 读取到的特征集数据批
        """
        pass

    @abstractmethod
    def read_lb_fn(self, index: Iterable and Sized,
                   n_worker: int = 1, pbar=None) -> Iterable:
        """加载标签集数据批所用方法

        :param index: 加载标签集数据批所用索引
        :param n_worker: 使用的处理机数目，若>1，则开启多线程处理
        :param pbar: 加载标签集时所用的进度条
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

    # @staticmethod
    # @abstractmethod
    # def wrap_fn(inputs: torch.Tensor,
    #             predictions: torch.Tensor,
    #             labels: torch.Tensor,
    #             metric_s: torch.Tensor,
    #             loss_es: torch.Tensor,
    #             comments: List[str]
    #             ):
    #     """输出结果打包函数
    #
    #     :param inputs: 预测所用到的输入批次数据
    #     :param predictions: 预测批次结果
    #     :param labels: 预测批次标签数据
    #     :param metric_s: 预测批次评价指标计算结果
    #     :param loss_es: 预测批次损失值计算结果
    #     :param comments: 预测批次每张结果输出图片附带注解
    #     :return: 打包结果
    #     """
    #     pass

    # @staticmethod
    # @abstractmethod
    # def save_fn(result, root: str) -> None:
    #     """输出结果存储函数
    #
    #     :param result: 输出结果图片批次
    #     :param root: 输出结果存储路径
    #     """
    #     pass

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

    def _set_wrap_fn(self, module: type):
        if hasattr(self, f'{module.__name__}_wrap_fn'):
            self.wrap_fn = getattr(self, f'{module.__name__}_wrap_fn')
        else:
            self.wrap_fn = self.default_wrap_fn

    def _set_preprocess(self, module: type):
        """
        根据处理模型的类型，自动指定预处理程序
        :param module: 用于处理本数据集的模型类型
        :return: None
        """
        if hasattr(self, f'{module.__name__}_preprocesses'):
            exec(f'self.{module.__name__}_preprocesses()')
        else:
            self.__default_preprocesses()

    # @abstractmethod
    def __default_preprocesses(self):
        """默认数据集预处理程序。
        注意：此程序均为本地程序，不可被序列化（pickling）！

        需要实现四个列表的填充：
        1. self.lbIndex_preprocesses：标签索引集预处理程序列表
        2. self.feaIndex_preprocesses：特征索引集预处理程序列表
        3. self.lb_preprocesses：标签集预处理程序列表
        4. self.fea_preprocesses：特征集预处理程序列表
        其中未指定本数据集为懒加载时，索引集预处理程序不会执行。

        若要定制其他网络的预处理程序，请定义函数：
        def {the_name_of_your_net}_preprocesses()
        """
        # 初始化预处理过程序
        self.lbIndex_preprocesses = toolz.compose()
        self.feaIndex_preprocesses = toolz.compose()
        self.lb_preprocesses = toolz.compose()
        self.fea_preprocesses = toolz.compose()

    def default_wrap_fn(self,
                        inputs: torch.Tensor,
                        predictions: torch.Tensor,
                        labels: torch.Tensor,
                        footnotes: list
                        ) -> Any:
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

    def __len__(self):
        return len(self._train_f)
