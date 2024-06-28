from torch.multiprocessing import Process as TORCHProcess
from torch.multiprocessing import Pipe


class Process(TORCHProcess):

    def __init__(self, target, *args, **kwargs):
        self.rec_conn, self.send_conn = Pipe(duplex=False)
        super().__init__(
            target=target,
            args=(self.send_conn, *args), kwargs=kwargs
        )
        self.exc = None

    def run(self):
        try:
            if self._target:
                self._target(*self._args, **self._kwargs)
                # super().start()
        except Exception as e:
            self.exc = e

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        if self.exc is not None:
            raise self.exc

    def get_result(self):
        return self.rec_conn.recv()
