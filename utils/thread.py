from threading import Thread as BuiltinThread


class Thread(BuiltinThread):

    def __init__(self, func, *args, **kwargs):
        """
        摘录自https://zhuanlan.zhihu.com/p/91601448
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.result = None
        super().__init__()

    def run(self):
        self.result = self.func(*self.args, **self.kwargs)

    def get_result(self):
        return self.result
