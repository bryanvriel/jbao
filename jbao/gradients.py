#-*- coding: utf-8 -*-

import jax
import jax.numpy as jnp

def vgrad(func, *args, n_inputs=1, argnum=1):
    """
    Convenience function for calling jax.vmap for a jax.grad computation in order to
    compute gradients for a batch of examples.
    """
    # Construct the in_axes tuple
    in_axes = (None,) * (len(args) - n_inputs) + (0,) * n_inputs
    #n_array_inputs = len(args) - 1
    #in_axes = (None,) + (0,) * n_array_inputs

    # Construct the vmap command
    vmapped_grad = jax.vmap(jax.grad(func, argnum), in_axes=in_axes, out_axes=0)

    # Call it
    value = vmapped_grad(*args)

    # Squeeze (in order to allow for subsequent gradients) and return
    return jnp.squeeze(value)

# end of file
