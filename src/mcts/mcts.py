from functools import partial

import jax
import jax.numpy as jnp

from src.config import MCTS_ITERS
from src.env.actions import clear_lines, step
from src.mcts.continuations import precompute_continuations
from src.mcts.rollout import rollout
from src.mcts.selection import select_batch
from src.mcts.tree import NUM_ACTIONS, NUM_CANDIDATES, init_candidates, update


@jax.jit
def q_value(tree):
    return tree["value_sum"] / (tree["visits"] + 1e-8)


@partial(jax.jit, static_argnames=["batch_size"])
def run_mcts(grid, root_tiles, batch_size=100):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    batch_size = min(batch_size, NUM_CANDIDATES)
    num_batches = (MCTS_ITERS + batch_size - 1) // batch_size
    tree = init_candidates(grid, root_tiles)
    continuations = precompute_continuations(grid, root_tiles)
    keys = jax.random.split(jax.random.key(0), num_batches)
    batch_ids = jnp.arange(num_batches)

    def body(tree, inputs):
        key, batch_id = inputs
        active = jnp.arange(batch_size) + batch_id * batch_size < MCTS_ITERS
        tree = mcts_iteration_batch(
            tree,
            continuations,
            key,
            active,
            batch_size,
        )
        return tree, None

    tree, _ = jax.lax.scan(body, tree, (keys, batch_ids))
    scores = jnp.where(tree["valid"], q_value(tree), -jnp.inf)
    best_idx = jnp.argmax(scores)
    tile_idx = best_idx // NUM_ACTIONS
    action = best_idx % NUM_ACTIONS
    result_grid, _, _ = step(grid, root_tiles[tile_idx], action)
    result_grid = clear_lines(result_grid)
    result_grid = jnp.where(tree["valid"][best_idx], result_grid, grid)

    return result_grid, tile_idx, action


@partial(jax.jit, static_argnames=["batch_size"])
def mcts_iteration_batch(tree, continuations, key, active, batch_size):
    indices, selected = select_batch(tree, key, batch_size)
    selected = selected & active
    tile_indices = indices // NUM_ACTIONS
    continuation_keys = jax.random.split(key, batch_size)
    choice_scores = jax.random.uniform(
        key,
        (batch_size, continuations["valid"].shape[1]),
    )
    choice_scores = jnp.where(
        continuations["valid"][tile_indices],
        choice_scores,
        -jnp.inf,
    )
    continuation_indices = jnp.argmax(choice_scores, axis=1)
    continuation_valid = jnp.any(continuations["valid"][tile_indices], axis=1)
    selected = selected & continuation_valid

    selected_grids = continuations["grids"][tile_indices, continuation_indices]
    values = jax.vmap(rollout)(selected_grids, continuation_keys)
    values = values + continuations["rewards"][tile_indices, continuation_indices]
    return update(tree, indices, values, selected)
