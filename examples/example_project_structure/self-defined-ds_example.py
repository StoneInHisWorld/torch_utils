import functools
import sys
from typing import Any, Generator

import toolz
import torch

from examples.example_project_structure.self_defined_ds_reader import YourDS_Reader
from examples.example_project_structure.self_defined_ds_transformer import YourDS_DataTransformer

sys.path.append('where/you/put/torch_utils')

from data_related.selfdefined_ds import SelfDefinedDataSet


"""此文件用于定义存储接触数据集，读取源数据"""


# TODO：自定义数据集
class TheNameOfYourDS(SelfDefinedDataSet):

    # TODO： 此处编写数据集路径检查逻辑
    def _check_path(self, root: str, which: str) -> [str, str, str, str]:
        """检查数据集路径是否正确，并生成和返回特征集和标签集路径

        :param root: 数据集所在根目录。
        :param which: 数据集名字。
        :return: 训练特征集目录，训练标签集目录，测试特征集目录，测试标签集目录
        """
        train_fd = ...
        train_ld = ...
        test_fd = ...
        test_ld = ...
        return train_fd, train_ld, test_fd, test_ld

    def _get_fea_index(self, root) -> list:
        """读取特征集索引
        本函数需要通过root来判断读取的是训练集还是测试集，同时root中需要传递数据集的存放位置

        :param root: 特征集路径
        :return: 特征索引集
        """
        fea_indices = []
        return fea_indices

    def _get_lb_index(self, root) -> list:
        """读取标签集索引
        本函数需要通过root来判断读取的是训练集还是测试集，同时root中需要传递数据集的存放位置

        :param root: 标签集路径
        :return: 标签索引集
        """
        lb_indices = []
        return lb_indices

    def _get_reader(self):
        """返回根据索引进行存储访问的数据读取器
        需要根据访问的数据类型，自定义数据读取器
        :return: 数据读取器
        """
        return YourDS_Reader()

    def _get_transformer(self):
        """返回执行数据预处理的数据转换器
        需要根据访问的数据类型以及任务需求，自定义数据预处理程序
        :return: 数据转换器
        """
        return YourDS_DataTransformer()

    @staticmethod
    def get_criterion_a():
        """指定训练过程中检验数据训练结果的评价指标

        :return: 评价指标计算函数列表
        """
        return [torch.nn.MSELoss()]

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
        raise NotImplementedError("未实现的函数")