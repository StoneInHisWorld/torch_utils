import functools
import multiprocessing
import time
from abc import abstractmethod
from typing import Iterable, Tuple, Sized, List, Callable, Any, Generator

import toolz
import torch
from tqdm import tqdm

from utils import itools, tstools, ControlPanel
from data_related.datasets import LazyDataSet, DataSet
from data_related.ds_operation import data_slicer
from utils.thread import Thread


class SelfDefinedDataSet:

    # 数据基本信息
    fea_channel = 1
    lb_channel = 1
    fea_mode = 'L'  # 读取二值图，读取图片速度将会大幅下降
    lb_mode = '1'
    f_req_sha = (256, 256)
    l_req_sha = (256, 256)

    def __init__(self, module: type, control_panel: dict or ControlPanel):
        """自定义DAO数据集
        负责按照用户指定的方式读取数据集的索引以及数据本身，并提供评价指标、数据预处理方法、结果包装方法。
        读取数据集的索引体现在_get_fea_index(), _get_lb_index()方法中，在此之前需要调用_check_path()检查数据集路径。
        读取数据体现在_get_fea_reader(), _get_lb_reader()方法中。
        评价指标通过get_criterion_a()方法提供。
        数据预处理方法通过AdoptedModelName_preprocesses()提供，请将AdoptedModelName替换为本次训练使用的模型类名
        结果包装方法通过AdoptedModelName_wrap_fn()提供，请将AdoptedModelName替换为本次训练使用的模型类名
        上述所有方法均需要用户自定义。

        :param module: 实验涉及数据集类型。数据集会根据实验所用模型来自动指定数据预处理程序以及结果包装程序。
        :param control_panel: 当前实验所属控制面板
        """
        # 从运行动态参数中获取参数
        where = control_panel['dataset_root']
        n_workers = control_panel['n_workers']
        data_portion = control_panel['data_portion']
        self.bulk_preprocess = control_panel['bulk_preprocess']
        shuffle = control_panel['shuffle']
        # 判断图片指定形状
        # self.f_required_shape = required_shape
        # self.l_required_shape = required_shape
        self.f_req_sha = tuple(control_panel['f_req_sha'])
        self.l_req_sha = tuple(control_panel['l_req_sha'])
        # 判断数据集懒加载程度
        # self._f_lazy = f_lazy
        # self._l_lazy = l_lazy
        self._f_lazy = self._l_lazy = control_panel['lazy']
        # 进行训练数据路径检查
        self._train_fd = None
        self._train_ld = None
        self._test_fd = None
        self._test_ld = None
        which = control_panel['which_dataset']
        self._check_path(where, which)
        print(f'\n{self.__class__.__name__}正在获取训练索引……')
        self._train_f, self._train_l = [], []
        self._get_fea_index(self._train_f, self._train_fd)
        self._get_lb_index(self._train_l, self._train_ld)
        # 按照数据比例切分数据集索引
        self._train_f, self._train_l = data_slicer(data_portion, shuffle, self._train_f, self._train_l)
        # 进行数据集加载
        if not self.bulk_preprocess:
            print(f'已选择单例预处理，将会在数据读取后立即进行预处理')
            self._set_preprocess(module)
        else:
            self.lb_preprocesses = None
            self.fea_preprocesses = None
        # 加载训练集
        indexes = ()
        if self._f_lazy:
            print('对于训练集，实行懒加载特征数据集')
        else:
            indexes = (self._train_f, )
        if self._l_lazy:
            print('对于训练集，实行懒加载标签数据集')
        else:
            indexes = (*indexes, self._train_l)
        self._train_f, self._train_l = self.read_fn(*indexes, n_workers)
        # 加载测试集
        print(f"\n{self.__class__.__name__}正在获取测试索引……")
        self._test_f, self._test_l = [], []
        self._get_fea_index(self._test_f, self._test_fd)
        self._get_lb_index(self._test_l, self._test_ld)
        self._test_f, self._test_l = data_slicer(data_portion, shuffle,
                                                 self._test_f, self._test_l)
        # 加载测试集
        if self._f_lazy:
            print('对于测试集，实行懒加载特征数据集')
        else:
            indexes = (self._test_f, )
        if self._l_lazy:
            print('对于测试集，实行懒加载标签数据集')
        else:
            indexes = (*indexes, self._test_l)
        self._test_f, self._test_l = self.read_fn(*indexes, n_workers)
        # 加载结果报告
        assert len(self._train_f) == len(self._train_l), \
            f'训练集的特征集和标签集长度{len(self._train_f)}&{len(self._train_l)}不一致'
        assert len(self._test_f) == len(self._test_l), \
            f'测试集的特征集和标签集长度{len(self._test_f)}&{len(self._test_l)}不一致'
        print("******数据集读取完毕******")
        print(f'数据集名称为{which}')
        print(f'数据集切片大小为{data_portion}')
        print(f'训练集长度为{len(self._train_f)}, 测试集长度为{len(self._test_f)}')
        print("************************")
        # 获取特征集、标签集及其索引集的预处理程序
        if self.bulk_preprocess:
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
        """检查数据集路径是否正确，否则直接中断程序。

        :param root: 数据集源目录。
        :param which: 数据集批次名
        """
        pass

    def read_fn(self,
                fea_index_or_d: Iterable and Sized, lb_index_or_d: Iterable and Sized,
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
        # fea_pbar = tqdm(
        #     total=len(fea_index_or_d), unit='张', position=0,
        #     desc=f"读取特征集图片中……", mininterval=1, leave=True, ncols=80
        # ) if not mute else None
        # read_fea_thread = Thread(
        #     self.read_fea_fn, fea_index_or_d, n_workers // 2, fea_pbar
        # )
        # read_fea_thread.start()
        # lb_pbar = tqdm(
        #     total=len(lb_index_or_d), unit='张', position=0,
        #     desc=f"读取标签集图片中……", mininterval=1, leave=True, ncols=80
        # ) if not mute else None
        # read_lb_thread = Thread(
        #     self.read_lb_fn, lb_index_or_d, n_workers // 2, lb_pbar
        # )
        # read_lb_thread.start()
        # if read_fea_thread.is_alive():
        #     read_fea_thread.join()
        # if read_lb_thread.is_alive():
        #     read_lb_thread.join()
        # fea_index_or_d = read_fea_thread.get_result()
        # lb_index_or_d = read_lb_thread.get_result()
        # return fea_index_or_d, lb_index_or_d

        def __read_impl(reader, indexes, preprocesses, n_workers, mute,
                        which='特征集'):
            if preprocesses:
                reader = toolz.compose(
                    preprocesses,
                    functools.partial(map, reader)  # 将读取单个索引的读取器转换为针对列表的读取器
                )
                # 将索引转化为压缩包内容，并升维以适应数据集级预处理程序
                indexes = [[index] for index in indexes]
            if int(n_workers) > 1:
                # 多进程读取
                with multiprocessing.Pool(n_workers) as p:
                    ret = p.map_async(reader, indexes)
                    start_time = time.perf_counter()
                    p.close()
                    i = 0
                    if not mute:
                        while not ret.ready():
                            print(f'\r{which}读取中，进行了{i}秒……', end='', flush=True)
                            time.sleep(1)
                            i += 1
                        print(f'\r{which}读取完毕，使用了{time.perf_counter() - start_time:.5f}秒', flush=True)
                    return ret.get()
            else:
                # 单进程读取
                if mute:
                    return list(map(reader, indexes))
                else:
                    return list(tqdm(
                        map(reader, indexes), total=len(indexes),
                        unit='张', desc=f'读取{which}图片中……', position=0,
                        mininterval=1, leave=True, ncols=80
                    ))

        threadpool = []
        if fea_index_or_d:
            # if self.fea_preprocesses:
            #     n_request = len(fea_index_or_d)
            # else:
            #     n_request = 1
            # fea_reader = next(self._get_fea_reader(n_request))
            fea_reader = next(self._get_fea_reader())
            read_fea_thread = Thread(
                __read_impl,
                fea_reader, fea_index_or_d, self.fea_preprocesses,
                n_workers // 2, mute
            )
            read_fea_thread.start()
            threadpool.append(read_fea_thread)
        if lb_index_or_d:
            # if self.lb_preprocesses:
            #     n_request = len(fea_index_or_d)
            # else:
            #     n_request = 1
            # lb_reader = next(self._get_lb_reader(n_request))
            lb_reader = next(self._get_lb_reader())
            read_lb_thread = Thread(
                __read_impl,
                lb_reader, lb_index_or_d, self.lb_preprocesses,
                n_workers // 2, mute, "标签集"
            )
            read_lb_thread.start()
            threadpool.append(read_lb_thread)
        for thread in threadpool:
            if thread.is_alive():
                thread.join()
        return [
            None if not fea_index_or_d else read_fea_thread.get_result(),
            None if not lb_index_or_d else read_lb_thread.get_result(),
        ]

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

    @staticmethod
    @abstractmethod
    def _get_fea_index(features, root) -> None:
        """读取根目录下的特征集索引

        :param features: 空列表对象
            请将所有的索引append()到本列表中
        :param root: 数据集根目录
        """
        pass

    @staticmethod
    @abstractmethod
    def _get_lb_index(labels, root) -> None:
        """读取根目录下的标签集索引

        :param labels: 空列表对象
            请将所有的索引append()到本列表中
        :param root: 数据集根目录
        """
        pass

    # @abstractmethod
    # # def read_fea_fn(self,
    # #                 indexes: Iterable and Sized,
    # #                 n_worker: int = 1, pbar: tqdm = None,
    # #                 preprocesses: Callable = None
    # #                 ) -> Iterable:
    # def read_fea_fn(self,
    #                 indexes: Iterable and Sized,
    #                 n_worker: int = 1, mute: bool = True,
    #                 preprocesses: Callable = None
    #                 ) -> Iterable:
    #     """加载特征集数据批所用方法
    #
    #     :param indexes: 加载特征集数据批所用索引
    #     :param n_worker: 使用的处理机数目，若>1，则开启多线程处理
    #     :param mute: 是否静默读取
    #     :param preprocesses: 预处理方法
    #         若指定了预处理方法，则
    #     :return:
    #     """
    #     """
    #
    #     :param indexes:
    #     :param n_worker:
    #     :param pbar: 加载特征集时所用的进度条
    #     :return: 读取到的特征集数据批
    #     """
    #     pass
    #
    # @abstractmethod
    # def read_lb_fn(self,
    #                indexes: Iterable and Sized,
    #                n_worker: int = 1, pbar: tqdm = None,
    #                preprocesses: Callable = None
    #                ) -> Iterable:
    #     """加载标签集数据批所用方法
    #
    #     :param indexes: 加载标签集数据批所用索引
    #     :param n_worker: 使用的处理机数目，若>1，则开启多线程处理
    #     :param pbar: 加载标签集时所用的进度条
    #     :return: 读取到的标签集数据批
    #     """
    #     pass

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
        test_ds.register_preprocess(features_calls=self.fea_preprocesses, labels_calls=self.lb_preprocesses)
        if self.bulk_preprocess:
            train_ds.preprocess(train_preprocess_desc)
            test_ds.preprocess(test_preprocess_desc)
        return train_ds, test_ds

    def _set_wrap_fn(self, module: type):
        """根据处理模型的类型，自动设置结果包装方法
        此函数会对self.wrap_fn进行赋值

        :param module: 用于处理本数据集的模型类型
        """
        if hasattr(self, f'{module.__name__}_wrap_fn'):
            self.wrap_fn = getattr(self, f'{module.__name__}_wrap_fn')
        else:
            self.wrap_fn = self.default_wrap_fn

    def _set_preprocess(self, module: type):
        """根据处理模型的类型，自动指定预处理程序
        此函数会对self.module_preprocesses进行赋值，若没有编写对应模型的预处理方法，则直接赋值为空函数。

        :param module: 用于处理本数据集的模型类型
        """
        if hasattr(self, f'{module.__name__}_preprocesses'):
            exec(f'self.{module.__name__}_preprocesses()')
        else:
            self.__default_preprocesses()

    def __default_preprocesses(self):
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

    def __len__(self):
        """数据集长度
        :return: 训练集长度，测试集长度
        """
        return len(self._train_f), len(self._test_f)
