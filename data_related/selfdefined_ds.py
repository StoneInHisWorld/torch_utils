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
# from utils import ControlPanel


class SelfDefinedDataSet:

    # # 数据基本信息
    # f_channel = 1
    # l_channel = 1
    # f_mode = 'L'
    # l_mode = '1'

    def __init__(self, module: type, is_train: bool, ds_config: dict, dl_config: dict):
        """自定义DAO数据集
        负责按照用户指定的方式读取数据集的索引以及数据本身，并根据模型类型提供评价指标。与StorageDataLoader相联系，调用数据读取方法；
            与DataTransformer相联系，调用数据预处理方法；与PredictionWrapper相联系，调用结果包装方法。
        在初始化函数__init__()函数中完成:
            1. 调用self._check_path()检查用户指定的"root/which"数据集文件中，数据路径是否满足用户要求。
            2. 调用self._get_fea_index(), self._get_lb_index()方法读取训练时所用的特征集索引以及标签集索引。验证集从这两个索引集中切分，
            3. 根据用户指定的懒加载状态，调用self.reader进行训练数据集读取
            4. 调用self._get_test_fea_index(), self._get_test_lb_index()方法读取测试时所用的特征集索引以及标签集索引。
            5. 根据用户指定的懒加载状态，调用self.reader进行测试数据集读取
        需要或者可选重载的方法包括：
            1. 数据集路径检查方法：_check_path()
            2. 训练特征集索引加载方法：_get_fea_index()
            3. 训练标签集索引加载方法：_get_lb_index()
            4. （可选）测试特征集索引加载方法：_get_test_fea_index()，默认调用self._get_fea_index()
            5. （可选）测试标签集索引加载方法：_get_test_lb_index()，默认调用self._get_lb_index()
            6. 指定数据读取的方法：_get_reader()
            7. 指定数据预处理器的方法：_get_transformer()
            8. 指定预测数据打包器的方法：get_wrapper()
            9. 涉及模型对应的评价指标方法：[ModelName]_criterion_a()
        评价指标通过get_criterion_a()方法提供。

        :param module: 实验涉及数据集类型。
        :param ds_config: 当前实验所属控制面板
        :param is_train: 是否处于训练模式。处于训练模式，则会加载训练和测试数据集，否则只提供测试数据集。
        """
        # 从运行动态参数中获取参数
        where = ds_config['dataset_root']
        self.n_workers = ds_config['n_workers']
        data_portion = ds_config['data_portion']
        self.bulk_preprocess = ds_config['bulk_preprocess']
        shuffle = ds_config['shuffle']
        # which = config['which_dataset']
        self.device = ds_config['device']
        # 判断图片指定形状
        self.f_req_shp = tuple(ds_config['f_req_shp']) if ds_config['f_req_shp'] else None
        self.l_req_shp = tuple(ds_config['l_req_shp']) if ds_config['l_req_shp'] else None
        # 指定数据集服务的模型类型
        self.module = module

        # 判断数据集懒加载程度
        self._f_lazy = self._l_lazy = ds_config['lazy']
        if self._f_lazy:
            print('对于训练集，实行懒加载特征数据集')
        if self._l_lazy:
            print('对于训练集，实行懒加载标签数据集')

        # 加载训练集
        self.is_train = is_train
        # 进行数据集路径检查
        directories = self._check_path(where)
        # 获取数据转换器和数据读取器
        self.transformer = self._get_transformer()
        self.reader = self._get_reader()
        if not self.bulk_preprocess:
            print(f'已选择单例预处理，将会在数据读取后立即进行预处理。本数据集将不会存储原始数据！')
            assert is_train, '测试模式下不允许单例预处理，否则将无法访问原始数据'
            if ds_config['lazy']:
                warnings.warn('懒加载和单例预处理冲突，优先选择单例预处理！')
                ds_config['lazy'] = False
        else:
            print("已选择批量预处理，将会在转化成数据迭代器前进行预处理")
        if is_train:
            print(f'\r{self.__class__.__name__}正在获取训练索引……', flush=True, end="")
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
        print(f"\r{self.__class__.__name__}正在获取测试索引……", flush=True, end="")
        # self._test_f, self._test_l = [fn(d) for fn, d in zip([self._get_fea_index, self._get_lb_index], directories)]
        self._test_f, self._test_l = [fn(d) for fn, d in zip(
            [self._get_test_fea_index, self._get_test_lb_index], directories
        )]
        self._test_f, self._test_l = data_slicer(data_portion, shuffle, self._test_f, self._test_l)
        self._test_f = self._test_f if self._f_lazy \
            else self.reader.fetch(self._test_f, None, not self.bulk_preprocess, False)[0]
        self._test_l = self._test_l if self._l_lazy \
            else self.reader.fetch(None, self._test_l, not self.bulk_preprocess, False)[0]

        # 加载结果报告
        assert len(self._test_f) == len(
            self._test_l), f'测试集的特征集和标签集长度{len(self._test_f)}&{len(self._test_l)}不一致'
        print("\n************数据集读取完毕*******************")
        # print(f'数据集名称为{which}')
        print(f'数据集切片大小为{data_portion}')
        if is_train:
            print(f'训练集选取长度为{len(self._train_f)}, 测试集选取长度为{len(self._test_f)}')
        else:
            print(f'测试集选取长度为{len(self._test_f)}')
        print("******************************************")
        self.ds_config = ds_config
        self.dl_config = dl_config

    @abstractmethod
    def _check_path(self, root: str) -> [str, str, str, str]:
        """检查数据集路径是否正确，并生成训练和测试数据的根目录
        如果发现数据集路径不正确则直接中断程序
        v0.5后不接受which参数，将有继承的子类在初始化函数中进行赋值，辅助路径判断

        :param root: 数据集源目录，指的是数据集文件存储的根目录
        :return 数据集路径。
            如果self.is_train == True，则返回[训练特征集路径，训练标签集路径，测试特征集路径，测试标签集路径]；
            否则返回[测试特征集路径，测试标签集路径]
        """
        raise NotImplementedError('没有编写路径检查程序！')

    @abstractmethod
    def _get_fea_index(self, root) -> list:
        """读取root根目录下的特征集索引，作为训练时的特征集。
        此处需要指定全量数据集的特征集索引存放目录。
        请不要在这里加载随机打乱的索引和进行数据集划分，这两项操作会由系统自动完成。

        :param root: 数据集根目录
        :return 提取出的特征集索引列表
        """
        raise NotImplementedError('没有编写特征集索引读取程序！')

    @abstractmethod
    def _get_lb_index(self, root) -> list:
        """读取root根目录下的标签集索引，作为训练时的标签集
        此处需要指定全量数据集的特征集索引存放目录。
        请不要在这里加载随机打乱的索引和进行数据集划分，这两项操作会由系统自动完成。

        :param root: 数据集根目录
        :return 提取出的标签集索引列表
        """
        raise NotImplementedError('没有编写标签集索引读取程序！')

    def _get_test_fea_index(self, root) -> list:
        """读取root根目录下的特征集索引，作为测试时的特征集。
        请不要在这里加载随机打乱的索引，这项操作会由系统自动完成。
        默认使用训练时的特征集读取方式，可以通过重载本方法进行自定义。

        :param root: 数据集根目录
        :return 提取出的特征集索引列表
        """
        return self._get_fea_index(root)

    def _get_test_lb_index(self, root) -> list:
        """读取根目录下的标签集索引，作为测试时的标签集。
        请不要在这里加载随机打乱的索引，这项操作会由系统自动完成。
        默认使用训练时的特征集读取方式，可以通过重载本方法进行自定义。

        :param root: 数据集根目录
        :return 提取出的标签集索引列表
        """
        return self._get_lb_index(root)

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
            raise NotImplementedError(f'{self.__class__.__name__}没有为{self.module.__name__}定义评价指标！'
                                      f'请在该类中定义{self.module.__name__}_criterion_a()来指定评价指标！')

    def refresh_preprocess(self):
        self.transformer.refresh()

    # def _to_dataset(self, i_cfn, collate_fn) -> list[LazyDataSet] or list[DataSet]:
    #     """根据自身模式，转换为合适的数据集，并对数据集进行预处理函数注册和执行。
    #     对于懒加载数据集，需要提供read_fn()，签名须为：
    #         read_fn(fea_index: Iterable[path], lb_index: Iterable[path]) -> Tuple[features: Iterable, labels: Iterable]
    #         数据加载器会自动提供数据读取路径index
    #     PS: 懒加载数据集生成方式不支持特征集或标签集的单独懒加载模式
    #
    #     :return: (训练数据集、测试数据集)，两者均为pytorch框架下数据集
    #     """
    #     # gen_datasets = []
    #     # fl_pairs = [(self._train_f, self._train_l), (self._test_f, self._test_l)] if self.is_train \
    #     #     else [(self._test_f, self._test_l)]
    #     # # TODO：这里的懒加载数据集生成方式不支持特征集或标签集的单独懒加载模式
    #     # if self._f_lazy or self._l_lazy:
    #     #     gen_datasets += [
    #     #         LazyDataSet(
    #     #             f, l, i_cfn, self.reader, self.transformer, collate_fn
    #     #         ) for f, l in fl_pairs
    #     #     ]
    #     #     train_preprocess_desc, test_preprocess_desc = \
    #     #         '\r正在对训练数据索引集进行预处理……', '\r正在对测试数据索引集进行预处理……'
    #     # else:
    #     #     # 生成数据集
    #     #     gen_datasets += [
    #     #         DataSet(f, l, self.transformer, collate_fn, self.device)
    #     #         for f, l in fl_pairs
    #     #     ]
    #     #     train_preprocess_desc, test_preprocess_desc = \
    #     #         '\r正在对训练数据集进行预处理……', '\r正在对测试数据集进行预处理……'
    #     # # 如进行批量预处理
    #     # if self.bulk_preprocess:
    #     #     for ds, desc in zip(gen_datasets, [train_preprocess_desc, test_preprocess_desc]):
    #     #         print(desc, flush=True)
    #     #         ds.preprocess()
    #     # # return gen_datasets
    #
    #     def __get_a_dataset(f, l, is_train):
    #         # TODO：这里的懒加载数据集生成方式不支持特征集或标签集的单独懒加载模式
    #         if self._f_lazy or self._l_lazy:
    #             ds = LazyDataSet(
    #                 i_cfn, self.reader,
    #                 f, l, self.transformer, collate_fn, is_train=is_train
    #             )
    #             desc = '训练数据索引集' if is_train else '测试数据索引集'
    #         else:
    #             # 生成数据集
    #             ds = DataSet(f, l, self.transformer, collate_fn, is_train=is_train, device=self.device)
    #             desc = '训练数据集' if is_train else '测试数据集'
    #         if self.bulk_preprocess:
    #             print(f"正在对{desc}进行预处理……", flush=True, end="")
    #             ds.preprocess(desc)
    #             print(f"\r{desc}预处理后长度为{len(ds)}")
    #         return ds
    #
    #     test_ds = __get_a_dataset(self._test_f, self._test_l, False)
    #     if self.is_train:
    #         train_ds = __get_a_dataset(self._train_f, self._train_l, True)
    #         return train_ds, test_ds
    #     return test_ds

    # def _to_dataset(self, i_cfn, collate_fn, transit_fn) -> list[LazyDataSet] or list[DataSet]:
    #     """根据自身模式，转换为合适的数据集，并对数据集进行预处理函数注册和执行。
    #     对于懒加载数据集，需要提供read_fn()，签名须为：
    #         read_fn(fea_index: Iterable[path], lb_index: Iterable[path]) -> Tuple[features: Iterable, labels: Iterable]
    #         数据加载器会自动提供数据读取路径index
    #     PS: 懒加载数据集生成方式不支持特征集或标签集的单独懒加载模式
    #
    #     :return: (训练数据集、测试数据集)，两者均为pytorch框架下数据集
    #     """
    #     non_blocking = self.config['non_blocking']
    #     share_memory = self.config['share_memory']
    #     transit_kwargs = self.config['transit_kwargs']
    #     device = self.config['device']
    #     bulk_transit = self.config['bulk_transit']
    #
    #     def __get_a_dataset(f, l, is_train):
    #         # TODO：这里的懒加载数据集生成方式不支持特征集或标签集的单独懒加载模式
    #         if self._f_lazy or self._l_lazy:
    #             raise NotImplementedError('懒加载模式还在维护当中')
    #             # ds = LazyDataSet(
    #             #     i_cfn, self.reader,
    #             #     f, l, self.transformer, collate_fn, is_train=is_train
    #             # )
    #             # desc = '训练数据索引集' if is_train else '测试数据索引集'
    #         else:
    #             # 生成数据集
    #             ds = DataSet(
    #                 f, l, self.transformer, is_train, bulk_transit, transit_fn,
    #                 non_blocking, share_memory, transit_kwargs, device, collate_fn
    #             )
    #             desc = '训练数据集' if is_train else '测试数据集'
    #         if self.bulk_preprocess:
    #             print(f"正在对{desc}进行预处理……", flush=True, end="")
    #             ds.preprocess(desc)
    #             print(f"\r{desc}预处理后长度为{len(ds)}")
    #         return ds
    #
    #     test_ds = __get_a_dataset(self._test_f, self._test_l, False)
    #     if self.is_train:
    #         train_ds = __get_a_dataset(self._train_f, self._train_l, True)
    #         return train_ds, test_ds
    #     return test_ds

    def _to_dataset(self, transit_fn) -> list[LazyDataSet] or list[DataSet]:
        """根据自身模式，转换为合适的数据集，并对数据集进行预处理函数注册和执行。
        对于懒加载数据集，需要提供read_fn()，签名须为：
            read_fn(fea_index: Iterable[path], lb_index: Iterable[path]) -> Tuple[features: Iterable, labels: Iterable]
            数据加载器会自动提供数据读取路径index
        PS: 懒加载数据集生成方式不支持特征集或标签集的单独懒加载模式

        :return: (训练数据集、测试数据集)，两者均为pytorch框架下数据集
        """
        non_blocking = self.ds_config['non_blocking']
        share_memory = self.ds_config['share_memory']
        transit_kwargs = self.ds_config['transit_kwargs']
        device = self.ds_config['device']
        bulk_transit = self.ds_config['bulk_transit']

        def __get_a_dataset(f, l, is_train):
            # TODO：这里的懒加载数据集生成方式不支持特征集或标签集的单独懒加载模式
            if self._f_lazy or self._l_lazy:
                raise NotImplementedError('懒加载模式还在维护当中')
                # ds = LazyDataSet(
                #     i_cfn, self.reader,
                #     f, l, self.transformer, collate_fn, is_train=is_train
                # )
                # desc = '训练数据索引集' if is_train else '测试数据索引集'
            else:
                # 生成数据集
                ds = DataSet(
                    f, l, self.transformer, is_train, bulk_transit, transit_fn,
                    non_blocking, share_memory, transit_kwargs, device
                )
                desc = '训练数据集' if is_train else '测试数据集'
            if self.bulk_preprocess:
                print(f"正在对{desc}进行预处理……", flush=True, end="")
                ds.preprocess(desc)
                print(f"\r{desc}预处理后长度为{len(ds)}")
            return ds

        test_ds = __get_a_dataset(self._test_f, self._test_l, False)
        if self.is_train:
            train_ds = __get_a_dataset(self._train_f, self._train_l, True)
            return train_ds, test_ds
        return test_ds

    # def to_dataloaders(self,
    #                    k, batch_size, i_cfn=None, collate_fn=None,
    #                    transit_fn: Callable = None, **dl_kwargs):
    #     """将自定义数据集转化为数据集迭代器
    #
    #     Args:
    #         k: 指定k折训练
    #         batch_size: 指定训练批量大小
    #         i_cfn: LazyDataSet所用索引整理函数
    #         collate_fn: torch.utils.data.DataLoader所用数据整理函数
    #         transit_fn: 数据迁移函数
    #         **dl_kwargs: torch.utils.data.DataLoader所用关键字参数
    #
    #     Returns:
    #         训练模式返回（训练数据迭代器，测试数据迭代器），测试模式返回测试数据迭代器
    #     """
    #     ret = []
    #     if self.is_train:
    #         train_ds, test_ds = self._to_dataset(i_cfn, collate_fn, transit_fn)
    #         assert 'train_portion' in dl_kwargs.keys(), "DataLoader参数缺少训练验证比'train_portion'！"
    #         train_portion = dl_kwargs.pop('train_portion')
    #         assert 'sampler' not in dl_kwargs.keys(), "请不要为DataLoader定制sampler，框架会自动生成！"
    #         # 使用k-fold机制
    #         data_iter_generator = (
    #             [
    #                 # dso.to_loader(train_ds, batch_size, transit_fn, sampler=sampler, **dl_kwargs)
    #                 train_ds.to_loader(batch_size, sampler=sampler, **dl_kwargs)
    #                 for sampler in sampler_group
    #             ]
    #             for sampler_group in dso.split_data(train_ds, k, train_portion)
    #         )  # 将抽取器遍历，构造加载器
    #         ret.append(data_iter_generator)
    #     else:
    #         if 'train_portion' in dl_kwargs.keys():
    #             warnings.warn("数据集的测试模式下train_portion参数将无效！")
    #             dl_kwargs.pop('train_portion')
    #         test_ds = self._to_dataset(i_cfn, collate_fn, transit_fn)
    #     # 获取测试集数据迭代器
    #     # test_iter = dso.to_loader(
    #     #     test_ds, batch_size, transit_fn, **dl_kwargs
    #     # )
    #     test_iter = test_ds.to_loader(batch_size, **dl_kwargs)
    #     return [*ret, test_iter]

    def to_dataloaders(self, transit_fn: Callable = None, **dl_kwargs):
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
        # 将框架的固定配置参数提取出来，合并到dl_kwargs中
        batch_size = self.dl_config.pop('batch_size')
        k = self.dl_config.pop('k')
        dl_kwargs.update(self.dl_config)
        # 根据训练模式进行数据集创建
        if self.is_train:
            train_portion = dl_kwargs.pop('train_portion')
            train_ds, test_ds = self._to_dataset(transit_fn)
            # assert 'train_portion' in dl_kwargs.keys(), "DataLoader参数缺少训练验证比'train_portion'！"
            assert 'sampler' not in dl_kwargs.keys(), "请不要为DataLoader定制sampler，框架会自动生成！"
            # 使用k-fold机制
            data_iter_generator = (
                [train_ds.to_loader(batch_size, sampler=sampler, **dl_kwargs)
                 for sampler in sampler_group]
                for sampler_group in dso.split_data(train_ds, k, train_portion)
            )  # 将抽取器遍历，构造加载器
            ret.append(data_iter_generator)
        else:
            if 'train_portion' in dl_kwargs.keys():
                warnings.warn("数据集的测试模式下train_portion参数将无效！")
                dl_kwargs.pop('train_portion')
            test_ds = self._to_dataset(transit_fn)
        # 获取测试集数据迭代器
        test_iter = test_ds.to_loader(batch_size, **dl_kwargs)
        return [*ret, test_iter]

    @abstractmethod
    def _get_transformer(self) -> DataTransformer:
        """告诉数据集，数据预处理的数据转换器
        需要根据访问的数据类型以及任务需求，自定义数据预处理程序
        :return: 数据转换器
        """
        raise NotImplementedError(f'{self.__class__.__name__}没有指定数据集转换器！')

    @abstractmethod
    def _get_reader(self) -> StorageDataLoader:
        """告诉数据集，根据索引进行存储访问的数据读取器
        该数据处理器为StorageDataLoader的子类

        :return: 数据读取器
        """
        raise NotImplementedError(f'{self.__class__.__name__}没有指定数据集读取程序！')

    @abstractmethod
    def get_wrapper(self, required_raw_ds=True) -> PredictionWrapper:
        """返回根据预测结果进行打包输出的结果打包器
        该结果打包器为PredictionWrapper的子类

        Args:
            required_raw_ds: 是否需要未进行预处理的原数据集

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
