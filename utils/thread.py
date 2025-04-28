from threading import Thread as BuiltinThread


class Thread(BuiltinThread):

    def __init__(self, func, *args, **kwargs):
        """重写的python内置线程，能够返回执行的结果
        摘录自https://zhuanlan.zhihu.com/p/91601448
        :param func: xuy
        :param args:
        :param kwargs:
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        self.exc = None
        super().__init__()

    def run(self):
        try:
            self.result = self.func(*self.args, **self.kwargs)
        except Exception as e:
            self.exc = e

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self.exc is not None:
            raise self.exc

    def get_result(self):
        if self.exc is not None:
            raise self.exc
        return self.result
