#-*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial
from typing import Sequence
import sys
import os


class DenseNetwork(nn.Module):

    layers: Sequence[int]
    actfun: str = 'tanh'

    @nn.compact
    def __call__(self, x):

        # Cache the activation function
        actfun = getattr(nn, self.actfun)

        # Loop over layers
        for cnt, layer in enumerate(self.layers[:-1]):
            x = actfun(nn.Dense(layer, name='layer%03d' % cnt)(x))
        x = nn.Dense(self.layers[-1], name='out_layer')(x)

        return x


class DenseResnet(nn.Module):

    layers: Sequence[int]
    actfun: str = 'tanh'

    @nn.compact
    def __call__(self, x):

        # Cache the activation function
        actfun = getattr(nn, self.actfun)

        # Loop over layers
        n_layers = len(self.layers)
        for cnt, layer in enumerate(self.layers[:-1]):
            z = actfun(nn.Dense(layer, name='layer%03d' % cnt)(x))
            if cnt > 0:
                x = x + z
            else:
                x = z
        x = nn.Dense(self.layers[-1], name='out_layer')(x)

        return x


class SpatialNetwork:
    """
    Specialized higher level dense network for taking in multiple spatial
    coordinates as inputs.
    """

    def __init__(self, n_layers, hidden_units, x_bounds, y_bounds,
                 input_dim=1, output_dim=1, name='model', **kwargs):

        # Save the name
        self.name = name

        # Store the input and output dimension sizes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_inputs = 2

        # Store coordinate bounds
        self.xmin, self.xmax = x_bounds
        self.ymin, self.ymax = y_bounds
        self.x_range = self.xmax - self.xmin
        self.y_range = self.ymax - self.ymin

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

    def apply(self, params, X, Y):

        # Normalize coordinates
        Xn = (X - self.xmin) / self.x_range
        Yn = (Y - self.ymin) / self.y_range

        # Pass to network and return
        return self.net.apply(params[self.name], Xn, Yn)

    # Make an alias to apply for streamlined usage
    def __call__(self, *args):
        return self.apply(*args)

    def dx(self, params, X, Y):
        gradfun = jax.grad(self.apply, 1)
        return jnp.squeeze(gradfun(params, X, Y))

    def dy(self, params, X, Y):
        gradfun = jax.grad(self.apply, 2)
        return jnp.squeeze(gradfun(params, X, Y))

    def dx2(self, params, X, Y):
        gradfun = jax.grad(self.dx, 1)
        return jnp.squeeze(gradfun(params, X, Y))

    def dy2(self, params, X, Y):
        gradfun = jax.grad(self.dy, 2)
        return jnp.squeeze(gradfun(params, X, Y))


# end of file
