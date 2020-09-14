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

    def __call__(self, *args):

        # Create model data
        data = []
        for n in range(self.N_layers):
            data.append(hk.Linear(self.Nh, w_init=self.w_init))
            data.append(self.actfun)
        data.append(hk.Linear(self.output_dim))

        # Create the sequential model
        model = hk.Sequential(data)

        # Optionally concatenate arguments
        if len(args) > 1:
            cargs = jnp.column_stack(args)
        else:
            cargs = args

        # Run prediction and squeeze outputs (for jax.grad compatibility)
        return jnp.squeeze(model(cargs))


class DenseNetwork(Model):
    """
    Higher level class for dense network.
    """

    def __init__(self, n_layers, hidden_units, input_dim=1, output_dim=1, n_inputs=1,
                 name='model', **kwargs):

        # Initialize base model class
        super().__init__(name=name)

        # Store the input and output dimension sizes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_inputs = n_inputs

        # Transform the network function
        self.net = hk.without_apply_rng(hk.transform(
            lambda *dargs: _DenseNetwork(output_dim, n_layers, hidden_units,
                                         name=name)(*dargs)
        ))

    def init_params(self, rng):
        """
        Randomly initialize parameters.
        """
        # Make inputs with correct shapes
        if self.n_inputs > 1:

            # Make list of inputs
            inputs = []
            for i in range(self.n_inputs):
                inputs.append(jnp.zeros((1, self.input_dim), dtype=jnp.float32))

            # Send to network init
            params = self.net.init(rng, *inputs)

        # Otherwise, send in a single input
        else:
            inputs = jnp.zeros((1, self.input_dim), dtype=jnp.float32)
            params = self.net.init(rng, inputs)

        return params


class SpatialNetwork(Model):
    """
    Specialized higher level dense network for taking in multiple spatial
    coordinates as inputs.
    """

    def __init__(self, n_layers, hidden_units, input_dim=1, output_dim=1,
                 name='model', **kwargs):

        # Initialize base model class
        super().__init__(name=name)

        # Store the input and output dimension sizes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_inputs = 2

        # Transform the network function
        self.net = hk.without_apply_rng(hk.transform(
            lambda *dargs: _DenseNetwork(output_dim, n_layers, hidden_units,
                                         name=name)(*dargs)
        ))

    def init_params(self, rng):
        """
        Randomly initialize parameters.
        """
        # Make list of inputs
        inputs = []
        for i in range(self.n_inputs):
            inputs.append(jnp.zeros((1, self.input_dim), dtype=jnp.float32))

        # Send to network init
        return self.net.init(rng, *inputs)

    #def dx(self, params, X, Y):
    #    dx = 1.0e-5
    #    fwd = self.net.apply(params[self.name], X + dx, Y)
    #    bwd = self.net.apply(params[self.name], X - dx, Y)
    #    return (fwd - bwd) / (2.0 * dx)

    #def dy(self, params, X, Y):
    #    dy = 1.0e-5
    #    fwd = self.net.apply(params[self.name], X, Y + dy)
    #    bwd = self.net.apply(params[self.name], X, Y - dy)
    #    return (fwd - bwd) / (2.0 * dy)

    def dx(self, params, X, Y):
        gradfun = jax.vmap(jax.grad(self.net.apply, 1), in_axes=(None, 0, 0), out_axes=0)
        return jnp.squeeze(gradfun(params[self.name], X, Y))

    def dy(self, params, X, Y):
        gradfun = jax.vmap(jax.grad(self.net.apply, 2), in_axes=(None, 0, 0), out_axes=0)
        return jnp.squeeze(gradfun(params[self.name], X, Y))


        

# end of file
