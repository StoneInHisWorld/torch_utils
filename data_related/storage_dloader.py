from concurrent.futures.thread import ThreadPoolExecutor
from typing import Callable

import toolz
from tqdm import tqdm


class StorageDataLoader:
    def __init__(
            self,
            f_reader: Callable, l_reader: Callable, transformer,
            n_workers: int = 1, mute: bool = False
    ):
        """访问存储的数据加载器"""
        self.__f_reader = f_reader
        self.__l_reader = l_reader
        self.transformer = transformer
        self.__fw_reader, self.__lw_reader = self.__wrap_reader()
        self.n_workers = n_workers
        self.mute = mute

    def fetch(self, fi, di, preprocess=False, n_workers=None):
        if n_workers is None:
            n_workers = self.n_workers
        else:
            n_workers = min(self.n_workers, n_workers)
        if preprocess:
            f_reader, l_reader = self.__fw_reader, self.__lw_reader
        else:
            f_reader, l_reader = self.__f_reader, self.__l_reader
        reading_args = []
        if fi:
            reading_args.append([f_reader, fi, "特征集"])
        if di:
            reading_args.append([l_reader, di, "标签集"])
        # 根据分配的处理机数目进行调用
        if n_workers < 2:
            return [self.__st_fetch(*args) for args in reading_args]
        elif n_workers < 4:
            return self.__dt_fetch(*reading_args)
        else:
            return self.__mt_fetch(*reading_args)

    def __st_fetch(self, reader, indices, which):
        """单线程"""
        ret = map(reader, indices)
        if not self.mute:
            return list(tqdm(
                ret, total=len(indices), position=0, leave=True,
                desc=f"\r正在读取{which}……", unit="个"
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
                    position=0, leave=True, desc=f"\r正在读取{which}……", unit="个"
                ))
            else:
                futures.append(xc.map(reader, indices))
        return [list(f) for f in futures]

    def __wrap_reader(self):
        assert hasattr(self, 'transformer'), '数据转换器还未定义'
        return [toolz.compose(
            lambda d: self.transformer.transform_data(d)[0],
            lambda d: [self.__f_reader(d)]  # 将读取单个索引的读取器转换为针对列表的读取器
        ), toolz.compose(
            lambda d: self.transformer.transform_data(None, d)[0],
            # functools.partial(self.transformer.transform_data, None),
            lambda d: [self.__l_reader(d)]  # 将读取单个索引的读取器转换为针对列表的读取器
        )]
