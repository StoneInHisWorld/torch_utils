class Accumulator:
    """浮点数累加器，负责训练过程中的浮点数指标的累加"""

    def __init__(self, n):
        """浮点数累加器
        负责训练过程中的浮点数指标的累加。
        :param n: 需要累加的指标数
        """
        self.data = [0.] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.] * len(self.data)

    def __getitem__(self, ids):
        return self.data[ids]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)
