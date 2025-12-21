from concurrent.futures.thread import ThreadPoolExecutor
from typing import Callable

import toolz
from tqdm import tqdm


class StorageDataLoader:

    def __init__(self, transformer, n_workers: int = 1, mute: bool = False,
                 f_reader: Callable = None, l_reader: Callable = None,
                 f_treader: Callable = None, l_treader: Callable = None):
        """访问存储的数据加载器
        用户需要继承本类，并在子类的__init__()方法中对数据加载办法f_reader以及l_reader进行自定义
        f_reader/f_treader的签名需要为def f_reader/f_treader(index) -> data，只需接收一个索引，返回一个样本数据
        l_reader/l_treader的签名需要为def l_reader/l_treader(index) -> data，只需接收一个索引，返回一个样本数据

        :param f_reader: 特征集的加载办法，需要为可调用对象
        :param l_reader: 标签集的加载办法，需要为可调用对象
        :param transformer: 数据集预处理器，用于对数据集进行单例预处理
        :param f_treader: 测试时特征集的加载办法，需要为可调用对象
        :param l_treader: 测试时标签集的加载办法，需要为可调用对象
        :param n_workers: 加载数据时所能使用的处理机数目
        :param mute: 加载时是否显示进度条
        """
        self.__f_reader = f_reader
        self.__l_reader = l_reader
        self.__f_treader = f_reader if f_treader is None else f_treader
        self.__l_treader = l_reader if l_treader is None else l_treader
        assert self.__f_treader is not None and self.__l_treader is not None, \
            "当测试集读取器缺省时，训练集读取器不可缺省！"
        self.transformer = transformer
        self.__fw_reader, self.__lw_reader, self.__fw_treader, self.__lw_treader = self.__wrap_reader()
        self.n_workers = n_workers
        self.mute = mute

    # def fetch(self, fi, di, preprocess=False, n_workers=None):
    #     if n_workers is None:
    #         n_workers = self.n_workers
    #     else:
    #         n_workers = min(self.n_workers, n_workers)
    #     if preprocess:
    #         f_reader, l_reader = self.__fw_reader, self.__lw_reader
    #     else:
    #         f_reader, l_reader = self.__f_reader, self.__l_reader
    #     reading_args = []
    #     if fi:
    #         reading_args.append([f_reader, fi, "特征集"])
    #     if di:
    #         reading_args.append([l_reader, di, "标签集"])
    #     # 根据分配的处理机数目进行调用
    #     if n_workers < 2:
    #         return [self.__st_fetch(*args) for args in reading_args]
    #     elif n_workers < 4:
    #         return self.__dt_fetch(*reading_args)
    #     else:
    #         return self.__mt_fetch(*reading_args)

    def fetch(self, fi=None, li=None, preprocess=False, is_train=True):
        if preprocess:
            # 如果需要预处理
            if is_train:
                # 获取训练时的，会进行单例预处理的数据集
                f_reader, l_reader = self.__fw_reader, self.__lw_reader
                prompt = "训练"
            else:
                # 获取测试时的，会进行单例预处理的数据集
                f_reader, l_reader = self.__fw_treader, self.__lw_treader
                prompt = "测试"
        elif is_train:
            # 获取训练时的数据集
            f_reader, l_reader = self.__f_reader, self.__l_reader
            prompt = "训练"
        else:
            # 获取测试时的数据集
            f_reader, l_reader = self.__f_treader, self.__l_treader
            prompt = "测试"
        reading_args = []
        if fi:
            reading_args.append([f_reader, fi, prompt + "特征集"])
        if li:
            reading_args.append([l_reader, li, prompt + "标签集"])
        # 根据分配的处理机数目进行调用
        if self.n_workers < 2:
            return [self.__st_fetch(*args) for args in reading_args]
        elif self.n_workers < 4:
            return self.__dt_fetch(*reading_args)
        else:
            return self.__mt_fetch(*reading_args)

    def __st_fetch(self, reader, indices, which):
        """单线程"""
        ret = map(reader, indices)
        if not self.mute:
            return list(tqdm(
                ret, total=len(indices), position=0, leave=True,
                desc=f"{self.__class__.__name__}正在读取{which}……", unit="个"
            ))
        else:
            return list(ret)

    def __dt_fetch(self, *args):
        """双线程"""
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(self.__st_fetch, *arg) for arg in args]
            return [f.result() for f in futures]

    def __mt_fetch(self, *args):
        """多线程"""
        xc1 = ThreadPoolExecutor(self.n_workers // 2)
        xc2 = ThreadPoolExecutor(self.n_workers // 2)
        futures = []
        for (reader, indices, which), xc in zip(args, [xc1, xc2]):
            if not self.mute:
                futures.append(tqdm(
                    xc.map(reader, indices), total=len(indices),
                    desc=f"{self.__class__.__name__}正在读取{which}……", unit="个"
                ))
            else:
                futures.append(xc.map(reader, indices))
        return [list(f) for f in futures]

    def __wrap_reader(self):
        """使用预处理转换器包装数据读取器

        :return: 会进行单例预处理的训练特征集读取器，会进行单例预处理的训练标签集读取器，
            会进行单例预处理的测试特征集读取器，会进行单例预处理的测试标签集读取器
        """
        assert hasattr(self, 'transformer'), f'{self.__class__.__name__}的数据转换器还未定义'
        return [toolz.compose(
            lambda d: self.transformer.transform_data(d)[0],
            lambda d: [self.__f_reader(d)]  # 将读取单个索引的读取器转换为针对列表的读取器
        ), toolz.compose(
            lambda d: self.transformer.transform_data(None, d)[0],
            lambda d: [self.__l_reader(d)]  # 将读取单个索引的读取器转换为针对列表的读取器
        ), toolz.compose(
            lambda d: self.transformer.transform_data(d, None, False)[0],
            lambda d: [self.__f_treader(d)]  # 将读取单个索引的读取器转换为针对列表的读取器
        ), toolz.compose(
            lambda d: self.transformer.transform_data(None, d, False)[0],
            lambda d: [self.__l_treader(d)]  # 将读取单个索引的读取器转换为针对列表的读取器
        )]
