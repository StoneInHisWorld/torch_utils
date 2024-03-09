class History:
    def __init__(self, *args: str):
        """
        历史记录器
        :param args: 指定记录项的名称
        """
        self.__keys = args
        for k in args:
            self.__setattr__(k, [])

    def __getitem__(self, key):
        return getattr(self, key)

    def add(self, keys, values):
        assert len(keys) == len(values)
        for k, v in zip(keys, values):
            self[k].append(v)

    def __iter__(self):
        for k in self.__keys:
            yield k, self[k]

    def __str__(self):
        ret = ''
        for k, v in self:
            ret += k + str(v)
        return ret

    def __iadd__(self, other):
        for k, h in other:
            assert hasattr(self, k), f'另一项历史记录有本项不记录的项{k}'
            self[k].extend(h)
        return self
