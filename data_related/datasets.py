from typing import Iterable, Callable, List

import dill
import torch
from torch.multiprocessing import Queue
from torch.utils.data import Dataset as torch_ds, DataLoader
from tqdm import tqdm

import utils.func.pytools as tools


# def fea_apply(result_pipe, features, features_calls, pbar_Q: Queue):
#     features_calls = dill.loads(features_calls)
#     n_tasks = len(features_calls)
#     for i, call in enumerate(features_calls):
#         features = call(features)
#         # shared_value['fea'] = call(shared_value['fea'])
#         pbar_Q.put(f'特征集预处理完成{i + 1}/{n_tasks}')
#         pbar_Q.put(1)
#     pbar_Q.put(True)
#     features = features.share_memory_()
#     result_pipe.send(features)
#
#
# def lb_apply(result_pipe, labels, labels_calls, pbar_Q: Queue):
#     # self = dill.loads(self)
#     labels_calls = dill.loads(labels_calls)
#     n_tasks = len(labels_calls)
#     for i, call in enumerate(labels_calls):
#         labels = call(labels)
#         # shared_value['lb'] = call(shared_value['lb'])
#         pbar_Q.put(f'标签集预处理完成{i + 1}/{n_tasks}')
#         pbar_Q.put(1)
#     pbar_Q.put(True)
#     labels = labels.share_memory_()
#     result_pipe.send(labels)


class DataSet(torch_ds):

    def __init__(self, features, labels,
                 collate_fn: Callable = None):
        """
        普通数据集，存储数据实际内容供DataLoader进行读取。
        :param features: 数据特征集
        :param labels: 数据标签集
        :param collate_fn: 数据预处理方法。DataLoader取出数据后，使用此方法对数据进行预处理。
        """
        assert isinstance(features, Iterable) and isinstance(labels, Iterable)
        assert len(features) == len(labels), f'特征集长度{len(features)}与标签集长度{len(labels)}不等！'
        self._features = features
        self._labels = labels
        self.collate_fn = collate_fn
        self.fea_preprocesses = []
        self.lb_preprocesses = []

    def __getitem__(self, item):
        return self._features[item], self._labels[item]

    def __len__(self):
        return len(self._features)

    def to(self, device: torch.device) -> None:
        """
        将数据迁移到device中
        :param device: 迁移目标设备
        :return: None
        """
        assert isinstance(self._features, torch.Tensor), \
            f'数据集特征集数据未转换成张量，无法迁移到{device}中！请在数据读取步骤和预处理步骤将特征集转换成张量。'
        assert isinstance(self._features, torch.Tensor), \
            f'数据集标签集数据未转换成张量，无法迁移到{device}中！请在数据读取步骤和预处理步骤将标签集转换成张量。'
        self._features = self._features.to(device)
        self._labels = self._labels.to(device)

    def apply(self, features_calls: List[Callable[[torch.Tensor], torch.Tensor]] = None,
              labels_calls: List[Callable[[torch.Tensor], torch.Tensor]] = None,
              pbar: tqdm = None):
        """对数据调用某种方法，可用作数据预处理。

        :param features_calls: 需要对特征集调用的方法列表
        :param labels_calls: 需要对标签集调用的方法列表
        :param pbar: 用于通知的进度条，指定为None则静默进行
        """
        if features_calls is None:
            features_calls = []
        if labels_calls is None:
            labels_calls = []
        if pbar is not None:
            pbar.reset(len(features_calls) + len(labels_calls))

        def pbar_update(n):
            """进度条更新方法"""
            if pbar is not None:
                pbar.update(n)

        def fea_apply():
            for call in features_calls:
                self._features = call(self._features)
                pbar_update(1)

        def lb_apply():
            for call in labels_calls:
                self._labels = call(self._labels)
                pbar_update(1)

        tools.multi_process(
            2, True, '',
            (fea_apply, (), {}),
            (lb_apply, (), {})
        )

        if pbar is not None:
            pbar.close()

    def to_loader(self, batch_size: int = None, shuffle=True,
                  sampler: Iterable = None, preprocess: bool = True,
                  **kwargs) -> DataLoader:
        """
        生成普通数据集的加载器，生成加载器前，会将所有预处理方法抛弃。
        :param preprocess: 是否进行预处理
        :param sampler: 实现了__len__()的可迭代对象，用于供给下标。若不指定，则使用默认sampler.
        :param batch_size: 每次供给的数据量。默认为整个数据集
        :param shuffle: 是否打乱
        :param kwargs: DataLoader额外参数
        :return: 加载器对象
        """
        if sampler is not None:
            shuffle = None
        if not batch_size:
            batch_size = self.feature_shape[0]
        if preprocess:
            self.preprocess()
            self.pop_preprocesses()
        return DataLoader(
            self, batch_size, shuffle=shuffle, collate_fn=self.collate_fn, sampler=sampler, **kwargs
        )

    def get_subset(self, indices: Iterable):
        return DataSet(self[indices][0], self[indices][1])

    def register_preprocess(self, features_calls=None, labels_calls=None):
        """
        注册预处理方法，用于数据加载器对数据进行预处理
        :param features_calls: 需要对特征集调用的方法列表
        :param labels_calls: 需要对标签集调用的方法列表
        """
        if features_calls is None:
            features_calls = []
        if labels_calls is None:
            labels_calls = []
        self.fea_preprocesses += features_calls
        self.lb_preprocesses += labels_calls

    def pop_preprocesses(self):
        """弹出所有的预处理程序。
        此方法用于对Dataset的序列化，将所有预处理的本地化方法转移到内存中。
        """
        try:
            del self.fea_preprocesses, self.lb_preprocesses
        except AttributeError:
            return

    def preprocess(self, desc='对数据集进行操作……'):
        """数据集对持有的特征集和标签集进行预处理，此为显式方法，会打印进度条"""
        pbar = tqdm(
            [], desc=desc, unit='步', position=0, mininterval=1,
            ncols=80
        )
        self.apply(self.fea_preprocesses, self.lb_preprocesses, pbar)

    @property
    def feature_shape(self):
        return self._features.shape[1:]

    @property
    def label_shape(self):
        return self._labels.shape[1:]


class LazyDataSet(DataSet):

    def __init__(self, features, labels, read_fn, collate_fn=None):
        """
        懒加载数据集，只存储数据的索引供LazyDataLoader使用。
        LazyDataLoader取该数据集中实际的数据内容时，会使用`read_fn`方法进行数据内容的读取。
        :param features: 数据特征集
        :param labels: 数据标签集
        :param read_fn: 数据读取方法。
            签名必须为：
            read_fn(fea_index: Iterable[path], lb_index: Iterable[path]) -> Tuple[features: Iterable, labels: Iterable]
            数据加载器会自动提供数据读取路径index
        :param collate_fn: 数据验证方法，签名需为：签名为List[T] -> Any. DataLoader取出数据后，使用此方法对数据进行验证。
        """
        self.read_fn = read_fn
        self.feaIndex_preprocess = []
        self.lbIndex_preprocess = []
        super().__init__(features, labels, collate_fn=collate_fn)

    def to_loader(self,
                  batch_size: int = None, sampler: Iterable = None, shuffle=True,
                  mute=False, **kwargs) -> DataLoader:
        """
        注意：本函数只会生成普通数据加载器。
        如生成懒数据加载器，则需要调用data_related.to_loader()
        :param sampler: 实现了__len__()的可迭代对象，用于供给下标。若不指定，则使用默认sampler.
        :param batch_size: 每次供给的数据量。默认为整个数据集
        :param shuffle: 是否打乱
        :param kwargs: DataLoader额外参数
        :return: 加载器对象
        """
        self.preprocess(mute=mute)
        return super().to_loader(
            batch_size, shuffle, sampler, preprocess=False, **kwargs
        )

    def register_preprocess(self, features_calls=None, labels_calls=None,
                            feaIndex_calls=None, lbIndex_calls=None):
        """
        注册预处理方法，用于数据加载器对数据进行预处理
        :param lbIndex_calls: 对于标签索引集的预处理方法
        :param feaIndex_calls: 对于特征索引集的预处理方法
        :param features_calls: 需要对特征集调用的方法列表
        :param labels_calls: 需要对标签集调用的方法列表
        :return: None
        """
        if feaIndex_calls is None:
            feaIndex_calls = []
        if lbIndex_calls is None:
            lbIndex_calls = []
        self.feaIndex_preprocess += feaIndex_calls
        self.lbIndex_preprocess += lbIndex_calls
        super().register_preprocess(features_calls, labels_calls)

    def pop_preprocesses(self):
        """弹出预处理特征集、标签集及其索引集的方法"""
        del self.feaIndex_preprocess, self.lbIndex_preprocess
        super().pop_preprocesses()

    def preprocess(self, desc='对懒加载数据集进行操作……', mute=False):
        """这里只对索引进行预处理

        :param desc: 预处理进度条描述
        :param mute: 是否静默处理
        """
        pbar = tqdm(
            [], desc=desc, unit='步', position=0, mininterval=1,
            ncols=80
        ) if not mute else None
        self.apply(
            self.feaIndex_preprocess, self.lbIndex_preprocess, pbar=pbar
        )

    def to(self, device: torch.device) -> None:
        """
        将数据迁移到device中
        :param device: 迁移目标设备
        :return: None
        """
        to = lambda fea: fea.to(device)
        if isinstance(self._features, torch.Tensor):
            self._features = self._features.to(device)
        elif to not in self.fea_preprocesses:
            self.fea_preprocesses += [to]
        if isinstance(self._labels, torch.Tensor):
            self._labels = self._labels.to(device)
        elif to not in self.lb_preprocesses:
            self.lb_preprocesses += [to]
