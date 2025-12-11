import toolz

from data_related.storage_dloader import StorageDataLoader


class YourDS_Reader(StorageDataLoader):

    def __init__(self, *args, **kwargs):
        """访问存储的数据集读取器，通过reader对象进行具体的读取逻辑编写
        f_reader与l_reader均为可调用对象
        """
        # 获取数据读取器
        f_reader = toolz.compose(*reversed([
        ]))
        l_reader = toolz.compose(*reversed([
        ]))
        super().__init__(f_reader, l_reader, **kwargs)

    def __del__(self):
        """可以在此处定义资源回收"""
        pass
