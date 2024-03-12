#-*- coding: utf-8 -*-

from typing import Callable

import jax
import jax.numpy as jnp

from jaxtyping import Array, Float, Int, PRNGKeyArray
import equinox as eqx

import sys
import os


class DenseNetwork(eqx.Module):

    layers: list
    activation: Callable = jax.nn.elu

    def __init__(
        self,
        layer_dims: list,
        input_dim: Int,
        output_dim: Int,
        key: PRNGKeyArray,
        activation: str = 'elu',
    ):
        # Fill out input and output layer dims
        layer_dims = [input_dim,] + layer_dims + [output_dim,]

        # Create PRNG keys for each layer
        n_layers = len(layer_dims) - 1
        keys = jax.random.split(key, n_layers)

        # Create list of linear layers
        self.layers = []
        for i in range(n_layers):
            self.layers.append(eqx.nn.Linear(
                in_features=layer_dims[i],
                out_features=layer_dims[i + 1],
                key=keys[i],
            ))

        # Create activation function
        self.activation = getattr(jax.nn, activation)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class DenseResNet(eqx.Module):

    layers: list
    activation: Callable = jax.nn.elu

    def __init__(
        self,
        layer_dims: list,
        input_dim: Int,
        output_dim: Int,
        key: PRNGKeyArray,
        activation: str = 'elu',
    ):
        # Fill out input and output layer dims
        layer_dims = [input_dim,] + layer_dims + [output_dim,]

        # Create PRNG keys for each layer
        n_layers = len(layer_dims) - 1
        keys = jax.random.split(key, n_layers)

        # Create list of linear layers
        self.layers = []
        for i in range(n_layers):
            self.layers.append(eqx.nn.Linear(
                in_features=layer_dims[i],
                out_features=layer_dims[i + 1],
                key=keys[i],
            ))

        # Create activation function
        self.activation = getattr(jax.nn, activation)

    def __call__(self, x):
        # Initial layer
        x = self.layers[0](x)
        # Intermediate layers with residual connection
        for layer in self.layers[1:-1]:
            z = self.activation(layer(x))
            x = x + z
        # Final layer
        return self.layers[-1](x)


# end of file
