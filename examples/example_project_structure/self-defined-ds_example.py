import sys
from typing import Iterable, Any, Sized

import torch
from PIL.Image import Image

sys.path.append('where/you/put/torch_utils')

from data_related.selfdefined_ds import SelfDefinedDataSet

"""此文件用于定义存储接触数据集，读取源数据"""


class TheNameOfYourDS(SelfDefinedDataSet):
    """TODO：自定义数据集"""

    def _check_path(self, root: str, which: str) -> None:
        """检查数据集路径是否正确，否则直接中断程序。
        TODO：此处编写数据集路径检查逻辑

        :param root: 数据集所在根目录。
        :param which: 数据集名字。
        :return: None
        """
        ...

    @staticmethod
    def _get_fea_index(features, root) -> None:
        """读取特征集索引

        :param features: 存储特征索引指针
        :param root: 特征集路径
        :return: None
        """
        ...

    @staticmethod
    def _get_lb_index(labels, root) -> None:
        """读取压缩包中的标签集索引

        :param labels: 存储标签索引指针
        :param root: 标签集路径
        :return: None
        """
        ...

    @staticmethod
    def read_fea_fn(indexes: Iterable and Sized, n_worker: int = 1) -> Iterable:
        """加载特征集数据所用方法

        :param n_worker: 使用的处理机数目，若>1，则开启多线程处理
        :param indexes: 加载特征集数据所用索引
        :return: 读取到的特征集数据
        """
        ...

    @staticmethod
    def read_lb_fn(indexes: Iterable and Sized, n_worker: int = 1) -> Iterable:
        """加载标签集数据批所用方法

        :param n_worker: 使用的处理机数目，若>1，则开启多线程处理
        :param indexes: 加载标签集数据所用索引
        :return: 读取到的标签集数据
        """
        ...

    @staticmethod
    def get_criterion_a():
        """指定训练过程中检验数据训练结果的评价指标

        :return: 评价指标计算函数列表
        """
        return [...]

    @staticmethod
    def wrap_fn(inputs: torch.Tensor,
                predictions: torch.Tensor,
                labels: torch.Tensor,
                metric_s: torch.Tensor,
                loss_es: torch.Tensor,
                comments: list
                ) -> Any:
        """输出结果打包函数

        :param inputs: 预测所用到的输入批次数据
        :param predictions: 预测批次结果
        :param labels: 预测批次标签数据
        :param metric_s: 预测批次评价指标计算结果
        :param loss_es: 预测批次损失值计算结果
        :param comments: 预测批次每张结果输出图片附带注解
        :return: 打包结果
        """
        ...

    @staticmethod
    def save_fn(result: Iterable[Image], root: str) -> None:
        """输出结果存储函数

        :param result: 输出结果图片批次
        :param root: 输出结果存储路径
        """
        ...

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
        ...
