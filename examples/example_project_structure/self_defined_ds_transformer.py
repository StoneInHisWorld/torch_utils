import toolz

from data_related.data_transformer import DataTransformer


class YourDS_DataTransformer(DataTransformer):

    def __init__(self, *args, **kwargs):
        """数据预处理器
        注意：此对象很有可能包括局部函数，因此不可被序列化！

        需要实现四个调用对象的编写
        1. self.li_preprocesses：标签索引集预处理程序
        2. self.fi_preprocesses：特征索引集预处理程序
        3. self.l_preprocesses：标签集预处理程序
        4. self.f_preprocesses：特征集预处理程序
        其中索引集预处理程序若未指定本数据集为懒加载，则不会执行。
        """
        super().__init__(*args, **kwargs)

    def UsedModel_preprocesses(self):
        """针对使用的网络类型，编写预处理程序
        需要将UsedModel替换成所用模型的类名，方可调用
        """
        # 设置数据索引集预处理程序
        self.fi_preprocesses = toolz.compose(*reversed([
        ]))
        self.li_preprocesses = toolz.compose(*reversed([
        ]))
        self.f_preprocesses = toolz.compose(*reversed([
        ]))
        self.l_preprocesses = toolz.compose(*reversed([
        ]))