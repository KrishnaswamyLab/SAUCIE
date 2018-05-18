from utils import *

class Loader(object):
    """A loader class designed to help provide batches one at a time in random order during training, or in the same order all at once when evaluating results."""

    def __init__(self, data, labels=None, shuffle=False):
        """
        Initialize Loader.

        :param data: array_like of size (N, D) for N points with D features
        :param labels: array_like of of size (N,) with a label for each point
        """
        self.start = 0
        self.epoch = 0
        self.data = [x for x in [data, labels] if x is not None]
        self.input_dim = data.shape[1]

        if shuffle:
            self.r = list(range(data.shape[0]))
            np.random.shuffle(self.r)
            self.data = [x[self.r] for x in self.data]

    def next_batch(self, batch_size=400):
        """
        Get the next batch of size batch_size.

        :param batch_size: the number of points to get
        """
        num_rows = self.data[0].shape[0]

        if self.start + batch_size < num_rows:
            batch = [x[self.start:self.start + batch_size] for x in self.data]
            self.start += batch_size
        # if we're at the end of data, wrap around and get some from the beginning
        else:
            self.epoch += 1
            batch_part1 = [x[self.start:] for x in self.data]
            batch_part2 = [x[:batch_size - (x.shape[0] - self.start)] for x in self.data]
            batch = [np.concatenate([x1, x2], axis=0) for x1, x2 in zip(batch_part1, batch_part2)]

            self.start = batch_size - (num_rows - self.start)

        return batch

    def iter_batches(self, batch_size=100):
        """
        Iterate through all the batches in the data.

        :param batch_size: the size of batch to yield each time as it's iterating
        """
        num_rows = self.data[0].shape[0]
        end = 0

        for i in range(num_rows // batch_size):
            start = i * batch_size
            end = (i + 1) * batch_size

            yield [x[start:end] for x in self.data]

        if end != num_rows:
            yield [x[end:] for x in self.data]

    def restore_order(self, data):
        """
        Since the data is randomly shuffled at initialization, this helper can return it to its original order if necessary.

        :param data: array_like of size (N,D)
        """
        data_out = np.zeros_like(data)
        for i, j in enumerate(self.r):
            data_out[j] = data[i]
        return data_out

