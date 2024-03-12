#-*- coding: utf-8 -*-

from typing import Callable
import logging
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
import json
import equinox as eqx
import optax
from tqdm import tqdm
import pickle


def train(
    model: eqx.Module,
    loss_fn: Callable,
    data,
    optimizer: optax.GradientTransformation,
    opt_state,
    key: PRNGKeyArray,
    model_hyperparams: dict,
    n_epochs=1000,
    callback=lambda model: 0,
    ckptfile='checkpoints.eqx',
    logfile='log_train',
    ckpt_skip=10,
):
    """
    NOTE: the loss_fn operates on a batch of data, so it should have a vmap
    somewhere inside of it. Also, the loss should return a list of losses.
    """
    # Reset logging file
    logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO, force=True)

    # Function for summing up the individual losses
    def sum_losses(model, batch, key=key):
        losses = loss_fn(model, batch, key=key)
        return jnp.sum(losses), losses

    # Define the update function
    @eqx.filter_jit
    def update_step(model, opt_state, batch, key):
        v_g = eqx.filter_value_and_grad(sum_losses, has_aux=True)
        (loss_value, losses), grads = v_g(model, batch, key=key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, losses

    # Call update function to compile and get number of losses
    b = data.train_batch()
    _, _, losses = update_step(model, opt_state, b, key)
    n_loss = len(losses)

    # Loop over epochs
    train_epoch = np.zeros((n_epochs, n_loss))
    test_epoch = np.zeros((n_epochs, n_loss))
    for epoch in tqdm(range(n_epochs)):

        # Callback at the beginning of each epoch
        callback(model)

        # Loop over batches
        train_loss = np.zeros((data.n_batches, n_loss))
        for i in range(data.n_batches):
            b = data.train_batch()
            model, opt_state, train_loss[i, :] = update_step(model, opt_state, b, key)
            key, key1 = jax.random.split(key)

        # Compute mean train loss
        train_loss = np.mean(train_loss, axis=0).tolist()

        # Compute mean loss for a batch of test data
        b = data.test_batch()
        test_loss = eqx.filter_jit(loss_fn)(model, b, key=key1).tolist()
    
        # Reshuffle data
        data.reset_training()

        # Store in epoch arrays
        train_epoch[epoch, :] = train_loss
        test_epoch[epoch, :] = test_loss

        # Write stats to logfile
        out = '%d ' + '%15.10f ' * 2 * n_loss
        logging.info(out % tuple([epoch] + train_loss + test_loss))
        sys.stdout.flush()

        # Periodically save checkpoint
        if epoch > 0 and epoch % ckpt_skip == 0:
            save_model(ckptfile, model, model_hyperparams, opt_state)

    # Save final checkpoint
    save_model(ckptfile, model, model_hyperparams, opt_state)

    # Return updated states and training stats
    return model, opt_state, train_epoch, test_epoch


def save_model(
    filename: str,
    model: eqx.Module,
    hyperparams: dict,
    opt_state=None,
):
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

    if opt_state is not None:
        optfile = filename + ".opt"
        with open(optfile, "wb") as f:
            eqx.tree_serialise_leaves(f, opt_state)


def load_model(filename: str, make_fn: Callable, optimizer=None, hyperparams=None):
    with open(filename, "rb") as f:

        # Load the hyperparameters if not provided
        temp = json.loads(f.readline().decode())
        if hyperparams is None:
            hyperparams = temp

        if 'input_dim' in hyperparams:
            value = hyperparams['input_dim']
            del hyperparams['input_dim']
            hyperparams['image_dim'] = [int(np.sqrt(value)), int(np.sqrt(value))]
        if 'model' not in hyperparams:
            hyperparams['model'] = 'dense'

        # Make the model
        model = make_fn(key=jax.random.PRNGKey(0), **hyperparams)
        out_model = eqx.tree_deserialise_leaves(f, model)

    if optimizer is None:
        return out_model
    else:
        init_opt_state = optimizer.init(eqx.filter(out_model, eqx.is_array))
        optfile = filename + ".opt"
        with open(optfile, "rb") as f:
            opt_state = eqx.tree_deserialise_leaves(f, init_opt_state)
        
        return out_model, opt_state
            

# end of file
