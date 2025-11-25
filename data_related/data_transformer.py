from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor


def identity(x): return x


class DataTransformer:

    def __init__(self, which_module: type, n_workers=1):
        """数据转换器
        负责对数据集按照服务的模型进行预处理和数据增强
        重载def augment_data(self, f, l) -> f, l方法以实现数据增强！

        :param which_module: 用于处理本数据集的模型类型
        :param n_workers: 转换器能使用的处理机个数
        """
        self.module_type = which_module
        self.n_workers = n_workers
        self.refresh()

    # def _default_preprocesses(self) -> None:
    #     """默认数据集预处理程序。
    #     注意：此程序均为本地程序，不可被序列化（pickling）！
    #
    #     类可定制化的预处理程序，需要以单个样本需要实现四个列表的填充：
    #     1. self.lbIndex_preprocesses：标签索引集预处理程序列表
    #     2. self.feaIndex_preprocesses：特征索引集预处理程序列表
    #     3. self.lb_preprocesses：标签集预处理程序列表
    #     4. self.fea_preprocesses：特征集预处理程序列表
    #     其中未指定本数据集为懒加载时，索引集预处理程序不会执行。
    #     请以数据集的角度（bulk_preprocessing）实现！
    #
    #     若要定制其他网络的预处理程序，请定义函数：
    #     def {the_name_of_your_net}_preprocesses()
    #     """
    #     # 初始化预处理过程序
    #     self.li_preprocesses = None
    #     self.fi_preprocesses = None
    #     self.l_preprocesses = None
    #     self.f_preprocesses = None

    def transform_indices(self, fi=None, li=None, is_train=True):
        # if n_workers is None:
        #     n_workers = self.n_workers
        # else:
        #     n_workers = min(self.n_workers, n_workers)
        futures = []
        if is_train:
            fi_preprocesses, li_preprocesses = self.fi_preprocesses, self.li_preprocesses
        else:
            fi_preprocesses, li_preprocesses = self.tfi_preprocesses, self.tli_preprocesses
        if self.n_workers > 1:
            with ThreadPoolExecutor(self.n_workers) as pool:
                if fi:
                    futures.append(pool.submit(fi_preprocesses, fi))
                if li:
                    futures.append(pool.submit(li_preprocesses, li))
            return [f.result() for f in futures]
        if fi:
            futures.append(fi_preprocesses(fi))
        if li:
            futures.append(li_preprocesses(li))
        return futures

    def transform_data(self, fea=None, lb=None, is_train=True):
        futures = []
        if is_train:
            f_preprocesses, l_preprocesses = self.f_preprocesses, self.l_preprocesses
        else:
            f_preprocesses, l_preprocesses = self.tf_preprocesses, self.tl_preprocesses
        if self.n_workers > 1:
            with ThreadPoolExecutor(self.n_workers) as pool:
                if fea:
                    futures.append(pool.submit(f_preprocesses, fea))
                if lb:
                    futures.append(pool.submit(l_preprocesses, lb))
                for future in as_completed(futures):
                    if future.exception():
                        pool.shutdown(wait=False)  # 立即终止线程池
                        raise future.exception()
            return [f.result() for f in futures]
        if fea:
            futures.append(f_preprocesses(fea))
        if lb:
            futures.append(l_preprocesses(lb))
        return futures

    def augment_data(self, f, l):
        # print(f"没有为{self.module_type.__name__}训练时指定数据增强程序，将返回源数据。"
        #       f"若要指定数据增强方法，请在{self.__class__.__name__}类中重载def augment_data(self, f, l) -> f, l方法！")
        return f, l

    def refresh(self, *args, **kwargs):
        """刷新预处理程序，需要在每次预处理数据之前调用一次

        :param args: 预处理程序所用位置参数
        :param kwargs: 预处理程序所用关键字参数
        """
        # 将之前定义的程序清空
        for preprocesses in ['fi_preprocesses', 'li_preprocesses', 'f_preprocesses', "l_preprocesses",
                             'tfi_preprocesses', 'tli_preprocesses', 'tf_preprocesses', 'tl_preprocesses']:
            if hasattr(self, preprocesses):
                delattr(self, preprocesses)
        # 寻找用户定制的预处理程序
        if hasattr(self, f'{self.module_type.__name__}_preprocesses'):
            getattr(self, f"{self.module_type.__name__}_preprocesses")(*args, **kwargs)
        else:
            raise NotImplementedError(f"没有为{self.module_type.__name__}定制预处理程序！"
                                      f"请在{self.__class__.__name__}类中创建{self.module_type.__name__}_preprocesses()方法！")
        # 检查用户预处理程序是否定义完备
        for preprocesses, serve_for in zip(['fi_preprocesses', 'li_preprocesses'], ["特征集的索引", "标签集的索引"]):
            if not hasattr(self, preprocesses):
                print(f"没有为{self.module_type.__name__}指定训练时{serve_for}的预处理程序，"
                      f"请在{self.module_type.__name__}_preprocesses()中通过self.{preprocesses}指定。"
                      f"将使用恒等函数替代！")
                setattr(self, preprocesses, identity)
        for preprocesses, serve_for in zip(['f_preprocesses', "l_preprocesses"], ["特征集", "标签集"]):
            assert hasattr(self, preprocesses), \
                (f"没有为{self.module_type.__name__}训练时的{serve_for}指定预处理程序，"
                 f"请在{self.module_type.__name__}_preprocesses()中通过self.{preprocesses}进行指定！")
        # 若用户未指定测试时预处理程序，则用训练的预处理程序作替代
        for preprocesses, backup, serve_for in zip(
            ['tfi_preprocesses', 'tli_preprocesses', 'tf_preprocesses', 'tl_preprocesses'],
            [identity, identity, self.f_preprocesses, self.l_preprocesses],
            ["特征集的索引", "标签集的索引", "特征集", "标签集"]
        ):
            if not hasattr(self, preprocesses):
                print(f"没有为{self.module_type.__name__}指定测试时{serve_for}的预处理程序，"
                      f"请在{self.module_type.__name__}_preprocesses()中通过self.{preprocesses}指定。"
                      f"将使用self.{backup.__name__}替代！")
                setattr(self, preprocesses, backup)
        # assert hasattr(self, f'fi_preprocesses'), \
        #     (f"没有为{self.module_type.__name__}训练时的特征集的索引指定预处理程序，"
        #      f"请在{self.module_type.__name__}_preprocesses()中通过self.fi_preprocesses进行赋值！")
        # assert hasattr(self, f'li_preprocesses'), \
        #     (f"没有为{self.module_type.__name__}训练时的标签集的索引指定预处理程序，"
        #      f"请在{self.module_type.__name__}_preprocesses()中通过self.li_preprocesses进行赋值！")
        # assert hasattr(self, f'f_preprocesses'), \
        #     (f"没有为{self.module_type.__name__}训练时的特征集指定预处理程序，"
        #      f"请在{self.module_type.__name__}_preprocesses()中通过self.f_preprocesses进行赋值！")
        # assert hasattr(self, f'l_preprocesses'), \
        #     (f"没有为{self.module_type.__name__}训练时的标签集指定预处理程序，"
        #      f"请在{self.module_type.__name__}_preprocesses()中通过self.l_preprocesses进行赋值！")
        # if not hasattr(self, f'tfi_preprocesses'):
        #     warnings.warn(f"没有为{self.module_type.__name__}指定测试时的特征集索引的预处理程序，"
        #                   f"请在{self.module_type.__name__}_preprocesses()中通过self.tfi_preprocesses指定。"
        #                   f"将使用self.fi_preprocesses替代！")
        #     self.tfi_preprocesses = self.fi_preprocesses
        # if not hasattr(self, f'tli_preprocesses'):
        #     warnings.warn(f"没有为{self.module_type.__name__}指定测试时的标签集索引的预处理程序，"
        #                   f"请在{self.module_type.__name__}_preprocesses()中通过self.tli_preprocesses指定。"
        #                   f"将使用self.li_preprocesses替代！")
        #     self.tli_preprocesses = self.li_preprocesses
        # if not hasattr(self, f'tf_preprocesses'):
        #     warnings.warn(f"没有为{self.module_type.__name__}指定测试时的特征集预处理程序，"
        #                   f"请在{self.module_type.__name__}_preprocesses()中通过self.tf_preprocesses指定。"
        #                   f"将使用self.f_preprocesses替代！")
        #     self.tf_preprocesses = self.f_preprocesses
        # if not hasattr(self, f'tl_preprocesses'):
        #     warnings.warn(f"没有为{self.module_type.__name__}指定测试时的标签集预处理程序，"
        #                   f"请在{self.module_type.__name__}_preprocesses()中通过self.tl_preprocesses指定。"
        #                   f"将使用self.l_preprocesses替代！")
        #     self.tl_preprocesses = self.l_preprocesses
