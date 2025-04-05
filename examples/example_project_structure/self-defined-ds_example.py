import functools
import sys
from typing import Any, Generator

import toolz
import torch

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

    def _get_fea_reader(self) -> Generator:
        """根据根目录下的特征集索引进行存储访问的数据读取器

        读取器为可调用对象，参数为单个索引值，如：fea = reader(index)
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
        reader = functools.partial()
        yield reader
        del reader

    def _get_lb_reader(self) -> Generator:
        """根据根目录下的标签集索引进行存储访问的数据读取器

        读取器为可调用对象，参数为单个索引值，如：lb = reader(index)
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
        reader = functools.partial()
        yield reader
        del reader

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
        ...

    def TheNameOfYourModel_preprocesses(self) -> None:
        """数据集预处理程序，请在方法名中替换对应模型的类名，
        在使用该模型进行训练时，SelfDefinedDataSet.set_preprocess()会将调用本函数，从而完成预处理程序的赋值

        需要实现四个列表的填充：
        1. self.lbIndex_preprocesses：标签索引集预处理程序列表
        2. self.feaIndex_preprocesses：特征索引集预处理程序列表
        3. self.lb_preprocesses：标签集预处理程序列表
        4. self.fea_preprocesses：特征集预处理程序列表
        其中索引集预处理程序若未指定本数据集为懒加载，则不会执行。

        :return: None
        """
        # 设置数据索引集预处理程序
        # 每个预处理程序都需要能同时处理单个数据或者数据集，以便适应单例预处理和批量预处理
        self.feaIndex_preprocesses = toolz.compose(*reversed([
        ]))
        self.lbIndex_preprocesses = toolz.compose(*reversed([
        ]))
        # 设置数据集预处理程序
        self.fea_preprocesses = toolz.compose(*reversed([
        ]))
        self.lb_preprocesses = toolz.compose(*reversed([
        ]))
