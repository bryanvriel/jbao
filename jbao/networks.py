#-*- coding: utf-8 -*-

import haiku as hk
import jax
import jax.numpy as jnp
import sys
import os


class Model:
    """
    Base model class for all networks.
    """

    def __init__(self, name='model'):

        # Store the name
        self.name = name

        # Initialize net to None; must be set by derived classes
        self.net = None

    def __call__(self, params, *args):
        return self.net.apply(params[self.name], *args)


class ModelCollection:

    def __init__(self, models):

        self.models = {}
        for model in models:
            self.models[model.name] = model

    def init_params(self, rng=None):

        # Random initialization
        if rng is not None:
            params = {}
            for name, model in self.models.items():
                params[name] = model.init_params(rng=rng)
        else:
            raise ValueError('Must pass in random number generator.')

        return params

    def __getitem__(self, key):
        return self.models[key]


class _DenseNetwork(hk.Module):
    """
    Base dense feedforward class for creating layers and function for passing
    through networks.
    """

    def __init__(self, output_dim, n_layers, hidden_units,
                 actfun=jnp.tanh, init='lecun_normal', name='densenet'):

        # Initialize parent class
        super().__init__(name=name)

        # Store parameters
        self.N_layers = n_layers
        self.Nh = hidden_units
        self.output_dim = output_dim

        # Go ahead and create initializer
        if init == 'lecun_normal':
            self.w_init = hk.initializers.VarianceScaling(1.0, 'fan_in', 'truncated_normal')
        elif init == 'lecun_uniform':
            self.w_init = hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform')
        elif init == 'glorot_normal':
            self.w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'truncated_normal')
        elif init == 'glorot_uniform':
            self.w_init = hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform')
        elif init == 'he_normal':
            self.w_init = hk.initializers.VarianceScaling(2.0, 'fan_in', 'truncated_normal')
        elif init == 'he_uniform':
            self.w_init = hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform')
        else:
            raise ValueError('Unsupported initializer.')

        # Save activation function
        self.actfun = actfun

    def __call__(self, X):

        # Create model data
        data = []
        for n in range(self.N_layers):
            data.append(hk.Linear(self.Nh, w_init=self.w_init))
            data.append(self.actfun)
        data.append(hk.Linear(self.output_dim))

        # Create the sequential model
        model = hk.Sequential(data)

        # Run prediction
        return model(X)


class DenseNetwork(Model):
    """
    Higher level class for dense network.
    """

    def __init__(self, input_dim, output_dim, *args, name='model'):

        # Initialize base model class
        super().__init__(name=name)

        # Store the input and output dimension sizes
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Transform the network function
        self.net = hk.without_apply_rng(hk.transform(
            lambda *dargs: _DenseNetwork(output_dim, *args, name=name)(*dargs)
        ))

    def init_params(self, rng):
        """
        Randomly initialize parameters.
        """
        # Make inputs with right shape for dense network
        inputs = jnp.zeros((1, self.input_dim), dtype=jnp.float32)

        # Call initialization
        params = self.net.init(rng, inputs)
        return params


# end of file
