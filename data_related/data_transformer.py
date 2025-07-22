from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor


class DataTransformer:

    def __init__(self, which_module: type, n_workers=1):
        """数据转换器
        负责对数据集按照服务的模型进行预处理

        :param module: 用于处理本数据集的模型类型
        """
        self.module_type = which_module
        self.n_workers = n_workers
        self.refresh()

    def _default_preprocesses(self) -> None:
        """默认数据集预处理程序。
        注意：此程序均为本地程序，不可被序列化（pickling）！

        类可定制化的预处理程序，需要以单个样本需要实现四个列表的填充：
        1. self.lbIndex_preprocesses：标签索引集预处理程序列表
        2. self.feaIndex_preprocesses：特征索引集预处理程序列表
        3. self.lb_preprocesses：标签集预处理程序列表
        4. self.fea_preprocesses：特征集预处理程序列表
        其中未指定本数据集为懒加载时，索引集预处理程序不会执行。
        请以数据集的角度（bulk_preprocessing）实现！

        若要定制其他网络的预处理程序，请定义函数：
        def {the_name_of_your_net}_preprocesses()
        """
        # 初始化预处理过程序
        self.li_preprocesses = None
        self.fi_preprocesses = None
        self.l_preprocesses = None
        self.f_preprocesses = None

    def transform_indices(self, fi=None, li=None, n_workers=None):
        if n_workers is None:
            n_workers = self.n_workers
        else:
            n_workers = min(self.n_workers, n_workers)
        futures = []
        if n_workers > 1:
            with ThreadPoolExecutor(n_workers) as pool:
                if self.fi_preprocesses:
                    futures.append(pool.submit(self.fi_preprocesses, fi))
                if self.li_preprocesses:
                    futures.append(pool.submit(self.li_preprocesses, li))
            return [f.result() for f in futures]
        if fi:
            futures.append(self.fi_preprocesses(fi))
        if li:
            futures.append(self.li_preprocesses(li))
        return futures

    def transform_data(self, fea=None, lb=None, n_workers=None):
        if n_workers is None:
            n_workers = self.n_workers
        else:
            n_workers = min(self.n_workers, n_workers)
        futures = []
        if n_workers > 1:
            with ThreadPoolExecutor(n_workers) as pool:
                if self.f_preprocesses and fea:
                    futures.append(pool.submit(self.f_preprocesses, fea))
                if self.l_preprocesses and lb:
                    futures.append(pool.submit(self.l_preprocesses, lb))
                for future in as_completed(futures):
                    if future.exception():
                        pool.shutdown(wait=False)  # 立即终止线程池
                        raise future.exception()
            return [f.result() for f in futures]
        if self.f_preprocesses and fea:
            futures.append(self.f_preprocesses(fea))
        if self.l_preprocesses and lb:
            futures.append(self.l_preprocesses(lb))
        return futures

    def refresh(self, *args, **kwargs):
        """刷新预处理程序，需要在每次预处理数据之前调用一次

        :param args: 预处理程序所用位置参数
        :param kwargs: 预处理程序所用关键字参数
        """
        if hasattr(self, f'{self.module_type.__name__}_preprocesses'):
            getattr(self, f"{self.module_type.__name__}_preprocesses")(*args, **kwargs)
            # exec(f'self.{self.module_type.__name__}_preprocesses()')
        else:
            self._default_preprocesses(*args, **kwargs)
