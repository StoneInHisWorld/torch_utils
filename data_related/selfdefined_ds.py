import warnings
from abc import abstractmethod
from typing import List, Callable

import torch

from data_related import ds_operation as dso
from data_related.data_transformer import DataTransformer
from data_related.datasets import LazyDataSet, DataSet
from data_related.ds_operation import data_slicer
from data_related.prediction_wrapper import PredictionWrapper
from data_related.storage_dloader import StorageDataLoader
from utils import ControlPanel


class SelfDefinedDataSet:
    # 数据基本信息
    f_channel = 1
    l_channel = 1
    f_mode = 'L'  # 读取二值图，读取图片速度将会大幅下降
    l_mode = '1'

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
        self.device = config['device']
        # 判断图片指定形状
        self.f_req_shp = tuple(config['f_req_shp']) if config['f_req_shp'] else None
        self.l_req_shp = tuple(config['l_req_shp']) if config['l_req_shp'] else None
        # 指定数据集服务的模型类型
        self.module = module

        # 判断数据集懒加载程度
        self._f_lazy = self._l_lazy = config['lazy']
        if self._f_lazy:
            print('对于训练集，实行懒加载特征数据集')
        if self._l_lazy:
            print('对于训练集，实行懒加载标签数据集')

        # 加载训练集
        self.is_train = is_train
        # 进行数据集路径检查
        directories = self._check_path(where, which)
        # 获取数据转换器和数据读取器
        self.transformer = self._get_transformer()
        self.reader = self._get_reader()
        if not self.bulk_preprocess:
            print(f'已选择单例预处理，将会在数据读取后立即进行预处理。本数据集将不会存储原始数据！')
            if config['lazy']:
                warnings.warn('懒加载和单例预处理冲突，优先选择单例预处理！')
                config['lazy'] = False
            assert is_train, '测试模式下不允许单例预处理，否则将无法访问原始数据'
        else:
            print("已选择批量预处理，将会在转化成数据迭代器前进行预处理")
        if is_train:
            print(f'{self.__class__.__name__}正在获取训练索引……')
            self._train_fd, self._train_ld = directories.pop(0), directories.pop(0)
            self._train_f, self._train_l = self._get_fea_index(self._train_fd), self._get_lb_index(self._train_ld)
            # 按照数据比例切分数据集索引
            self._train_f, self._train_l = data_slicer(data_portion, shuffle, self._train_f, self._train_l)
            self._train_f = self._train_f if self._f_lazy \
                else self.reader.fetch(self._train_f, None, not self.bulk_preprocess)[0]
            self._train_l = self._train_l if self._l_lazy \
                else self.reader.fetch(None, self._train_l, not self.bulk_preprocess)[0]
            assert len(self._train_f) == len(self._train_l), \
                f'训练集的特征集和标签集长度{len(self._train_f)}&{len(self._train_l)}不一致'
        # 加载测试集
        print(f"{self.__class__.__name__}正在获取测试索引……")
        self._test_f, self._test_l = [fn(d) for fn, d in zip([self._get_fea_index, self._get_lb_index], directories)]
        self._test_f, self._test_l = data_slicer(data_portion, shuffle, self._test_f, self._test_l)
        self._test_f = self._test_f if self._f_lazy \
            else self.reader.fetch(self._test_f, None, not self.bulk_preprocess)[0]
        self._test_l = self._test_l if self._l_lazy \
            else self.reader.fetch(None, self._test_l, not self.bulk_preprocess)[0]

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
        if not self.bulk_preprocess:
            # 整理读取和预处理过后的数据，并清空预处理程序
            self._train_f = torch.vstack(self._train_f)
            self._train_l = torch.vstack(self._train_l)
            self._test_f = torch.vstack(self._test_f)
            self._test_l = torch.vstack(self._test_l)

    @abstractmethod
    def _check_path(self, root: str, which: str) -> [str, str, str, str]:
        """检查数据集路径是否正确，否则直接中断程序。

        :param root: 数据集源目录。
        :param which: 数据集批次名
        :return 返回正确的路径名。
            如果self.is_train == True，则返回[训练特征集路径，训练标签集路径，测试特征集路径，测试标签集路径]；
            否则返回[测试特征集路径，测试标签集路径]
        """
        raise NotImplementedError('没有编写路径检查程序！')

    @abstractmethod
    def _get_fea_index(self, root) -> list:
        """读取根目录下的特征集索引

        :param root: 数据集根目录
        :return 提取出的索引列表
        """
        raise NotImplementedError('没有编写特征集索引读取程序！')

    @abstractmethod
    def _get_lb_index(self, root) -> list:
        """读取根目录下的标签集索引

        :param root: 数据集根目录
        :return 提取出的索引列表
        """
        raise NotImplementedError('没有编写标签集索引读取程序！')

    def get_criterion_a(self) -> List[
        Callable[
            [torch.Tensor, torch.Tensor, bool],
            float or torch.Tensor
        ]
    ]:
        """获取数据集的评价指标函数

        请注意，返回的指标函数不能有重名，否则可能会导致数据计算错误或者历史记录趋势图绘制重叠
        :return: 一系列指标函数。签名均为
        def criterion(Y_HAT, Y, size_averaged) -> float or torch.Tensor
        其中size_averaged表示是否要进行批量平均。
        """
        if hasattr(self, f'{self.module.__name__}_criterion_a'):
            return getattr(self, f'{self.module.__name__}_criterion_a')()
        else:
            return []

    def refresh_preprocess(self):
        self.transformer.refresh()

    def _to_dataset(self, i_cfn, collate_fn) -> list[LazyDataSet] or list[DataSet]:
        """根据自身模式，转换为合适的数据集，并对数据集进行预处理函数注册和执行。
        对于懒加载数据集，需要提供read_fn()，签名须为：
            read_fn(fea_index: Iterable[path], lb_index: Iterable[path]) -> Tuple[features: Iterable, labels: Iterable]
            数据加载器会自动提供数据读取路径index
        PS: 懒加载数据集生成方式不支持特征集或标签集的单独懒加载模式

        :return: (训练数据集、测试数据集)，两者均为pytorch框架下数据集
        """
        gen_datasets = []
        fl_pairs = [(self._train_f, self._train_l), (self._test_f, self._test_l)] if self.is_train \
            else [(self._test_f, self._test_l)]
        # TODO：这里的懒加载数据集生成方式不支持特征集或标签集的单独懒加载模式
        if self._f_lazy or self._l_lazy:
            gen_datasets += [
                LazyDataSet(
                    f, l, i_cfn, self.reader, self.transformer, collate_fn
                ) for f, l in fl_pairs
            ]
            train_preprocess_desc, test_preprocess_desc = \
                '\r正在对训练数据索引集进行预处理……', '\r正在对测试数据索引集进行预处理……'
        else:
            # 生成数据集
            gen_datasets += [
                DataSet(f, l, self.transformer, collate_fn, self.device)
                for f, l in fl_pairs
            ]
            train_preprocess_desc, test_preprocess_desc = \
                '\r正在对训练数据集进行预处理……', '\r正在对测试数据集进行预处理……'
        # 如进行批量预处理
        if self.bulk_preprocess:
            for ds, desc in zip(gen_datasets, [train_preprocess_desc, test_preprocess_desc]):
                print(desc, flush=True)
                ds.preprocess()
        return gen_datasets

    def to_dataloaders(self,
                       k, batch_size, i_cfn=None, collate_fn=None,
                       transit_fn: Callable = None, **dl_kwargs):
        """将自定义数据集转化为数据集迭代器

        Args:
            k: 指定k折训练
            batch_size: 指定训练批量大小
            i_cfn: LazyDataSet所用索引整理函数
            collate_fn: torch.utils.data.DataLoader所用数据整理函数
            transit_fn: 数据迁移函数
            **dl_kwargs: torch.utils.data.DataLoader所用关键字参数

        Returns:
            训练模式返回（训练数据迭代器，测试数据迭代器），测试模式返回测试数据迭代器
        """
        ret = []
        if self.is_train:
            train_ds, test_ds = self._to_dataset(i_cfn, collate_fn)
            train_portion = dl_kwargs.pop('train_portion', 0.8)
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
            ret.append(data_iter_generator)
        else:
            test_ds = self._to_dataset(i_cfn, collate_fn)[0]
        # 获取测试集数据迭代器
        test_iter = dso.to_loader(
            test_ds, batch_size, transit_fn, **dl_kwargs
        )
        return [*ret, test_iter]

    @abstractmethod
    def _get_transformer(self) -> DataTransformer:
        """返回执行数据预处理的数据转换器
        需要根据访问的数据类型以及任务需求，自定义数据预处理程序
        :return: 数据转换器
        """
        raise NotImplementedError('没有指定数据集转换器！')

    @abstractmethod
    def _get_reader(self) -> StorageDataLoader:
        """返回根据索引进行存储访问的数据读取器
        该数据处理器为StorageDataLoader的子类
        :return: 数据读取器
        """
        raise NotImplementedError('没有指定特征集读取程序！')

    @abstractmethod
    def get_wrapper(self, required_raw_ds=True) -> PredictionWrapper:
        """返回根据预测结果进行打包输出的结果打包器
        该结果打包器为PredictionWrapper的子类

        Args:
            required_raw_ds: 需要未进行预处理的原数据集

        Returns:
            结果打包器
        """
        raise NotImplementedError('没有指定结果打包器！')

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
