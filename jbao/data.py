#-*- coding: utf-8 -*-

import numpy as np
import sys
import os


def atleast_2d(x):
    """
    Convenience function to ensure arrays are column vectors.
    """
    if x.ndim == 1:
        return x.reshape(-1, 1)
    elif x.ndim == 2:
        return x
    else:
        raise NotImplementedError('Input array has greater than 2 dimensions')


def train_test_indices(N, train_fraction=0.9, shuffle=True, rng=None):
    """
    Convenience function to get train/test splits.
    """
    n_train = int(np.floor(train_fraction * N))
    if shuffle:
        assert rng is not None, 'Must pass in a random number generator'
        ind = rng.permutation(N)
    else:
        ind = np.arange(N, dtype=int)
    ind_train = ind[:n_train]
    ind_test = ind[n_train:]

    return ind_train, ind_test


class Data:
    """
    Class for representing and returning scattered points of solutions and coordinates.
    """

    def __init__(self, *args, train_fraction=0.9, batch_size=1024, shuffle=True,
                 seed=None, split_seed=None, **kwargs):
        """
        Initialize dictionary of data and batching options. Data should be passed in
        via the kwargs dictionary.
        """
        # Check nothing has been passed in *args
        if len(args) > 0:
            raise ValueError('Data does not accept non-keyword arguments.')

        # Create a random number generator
        self.rng = np.random.RandomState(seed=seed)
        if split_seed is not None:
            self.split_rng = np.random.RandomState(seed=split_seed)
        else:
            self.split_rng = np.random.RandomState(seed=seed)

        # Assume the coordinate T exists to determine data size
        self.shuffle = shuffle
        self.n_data = kwargs['T'].shape[0]
    
        # Generate train/test indices
        itrain, itest = train_test_indices(self.n_data,
                                           train_fraction=train_fraction,
                                           shuffle=shuffle,
                                           rng=self.split_rng)

        # Unpack the data for training
        self.keys = sorted(kwargs.keys())
        self._train = {}
        for key in self.keys:
            self._train[key] = kwargs[key][itrain]

        # Unpack the data for testing
        self._test = {}
        for key in self.keys:
            self._test[key] = kwargs[key][itest]

        # Cache training and batch size
        self.n_train = self._train['T'].shape[0]
        self.n_test = self.n_data - self.n_train
        self.batch_size = batch_size
        self.n_batches = int(np.ceil(self.n_train / self.batch_size))

        # Initialize training indices (data have already been shuffle, so only need arange here)
        self._itrain = np.arange(self.n_train, dtype=int)

        # Initialize counter for training data retrieval
        self._train_counter = 0

        return

    def train_batch(self):
        """
        Get a random batch of training data as a dictionary. Ensure that we cycle through
        complete set of training data (e.g., sample without replacement). NOTE: jax.jit
        needs to know batch sizes at compile time, so in order to minimize compilation
        time, we ensure that every train batch is the same batch size.
        """
        # If we've already reached the end of the training data, re-set counter with
        # optional re-shuffling of training indices
        if self._train_counter >= self.n_train:
            self._train_counter = 0
            if self.shuffle:
                self._itrain = self.rng.permutation(self.n_train)

        # Construct slice for training data indices
        islice = slice(self._train_counter, self._train_counter + self.batch_size)
        indices = self._itrain[islice]

        # Since we want to maintain a constant batch size, pad indices with extra
        # random sampling of training indices
        if indices.size < self.batch_size:
            n_pad = self.batch_size - indices.size
            rand_ind = self.rng.choice(self._itrain, size=n_pad, replace=False)
            indices = np.hstack((indices, rand_ind))

        # Get training data
        result = {key: self._train[key][indices] for key in self.keys}

        # Update counter for training data
        self._train_counter += self.batch_size

        # All done
        return result

    def test_batch(self, batch_size=None):
        """
        Get a random batch of testing data as a dictionary.
        """
        batch_size = batch_size or self.batch_size
        ind = self.rng.choice(self.n_test, size=batch_size)
        return {key: self._test[key][ind] for key in self.keys}

    @property
    def train(self):
        """
        Get entire training set.
        """
        return self._train

    @train.setter
    def train(self, value):
        raise ValueError('Cannot set train variable.')

    @property
    def test(self):
        """
        Get entire testing set.
        """
        return self._test

    @test.setter
    def test(self, value):
        raise ValueError('Cannot set test variable.')


class Normalizer:
    """
    Simple convenience class that performs transformations to/from normalized values.
    Here, we use the norm range [-1, 1] for pos=False or [0, 1] for pos=True.
    """

    def __init__(self, xmin, xmax, pos=False):
        self.xmin = xmin
        self.xmax = xmax
        self.denom = xmax - xmin
        self.pos = pos

    def __call__(self, x):
        """
        Alias for Normalizer.forward()
        """
        return self.forward(x)

    def forward(self, x):
        """
        Normalize data.
        """
        if self.pos:
            return (x - self.xmin) / self.denom
        else:
            return 2.0 * (x - self.xmin) / self.denom - 1.0

    def inverse(self, xn):
        """
        Un-normalize data.
        """
        if self.pos:
            return self.denom * xn + self.xmin
        else:
            return 0.5 * self.denom * (xn + 1.0) + self.xmin


# end of file
