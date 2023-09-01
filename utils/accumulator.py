class Accumulator:
    def __init__(self, n):
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
