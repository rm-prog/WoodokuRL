from functools import partial

import jax
import jax.numpy as jnp

from src.config import MCTS_ITERS
from src.env.actions import clear_lines, step
from src.mcts.rollout import rollout
from src.mcts.selection import select_batch, select
from src.mcts.expansion import expand
from src.mcts.tree import init_tree, update, update_node, add_child
from src.mcts.backprop import backprop

from src.env.tiles import TILE_PERMS

@jax.jit
def q_value(tree):
    return tree["value_sum"] / (tree["visits"] + 1e-8)

@jax.jit
def run_mcts(grid, root_tiles, tiles_available):
    tree = init_tree()
    root = 0

    tree = add_child(tree, -1, -1, 0, root, tiles_available)
    tree = update_node(tree, root, 0.0)

    keys = jax.random.split(jax.random.key(0), MCTS_ITERS)

    def body(tree, key):
        tree = mcts_iteration(tree, grid, root_tiles, key)
        return tree, None

    tree, _ = jax.lax.scan(body, tree, keys)
    scores = jnp.where((tree["value_sum"] > 0) & (tree["depth"] == 1), q_value(tree), -jnp.inf)
    any_valid = jnp.any(scores > -jnp.inf)
    best_idx = jnp.argmax(scores)
    best_action = tree["action_from_parent"][best_idx]

    return jax.lax.cond(
        any_valid,
        lambda _: best_action,
        lambda _: -1,
        operand=None)

@jax.jit
def mcts_iteration(tree, grid, root_tiles, key):
    idx, selected_grid = select(tree, grid, root_tiles)
    tree, chosen_child, chosen_grid = expand(tree, idx, selected_grid, root_tiles, key)
    available_tiles = tree["tiles_available"][chosen_child]
    value = rollout(chosen_grid, root_tiles, available_tiles, key)
    tree = backprop(tree, chosen_child, value)

    return tree

@jax.jit
def three_move_mcts(grid, tiles):

    available_tiles = jnp.ones(3, dtype=jnp.bool_)

    def one_mcts(grid, tiles, available_tiles, key):
        action = run_mcts(grid, tiles, available_tiles)

        def valid_path(_):
            tile_idx = action // 81
            position = action % 81
            new_grid, _, _ = step(grid, tiles[tile_idx], position)
            new_grid = clear_lines(new_grid)
            new_available_tiles = available_tiles.at[tile_idx].set(False)
            return new_grid, new_available_tiles, True

        def invalid_path(_):
            return grid, available_tiles, False

        grid, available_tiles, valid = jax.lax.cond(
            action >= 0,
            valid_path,
            invalid_path,
            operand=None
        )

        return grid, available_tiles, valid, action

    def body(carry, _):
        grid, available_tiles, key = carry
        key, subkey = jax.random.split(key)
        grid, available_tiles, valid, action = one_mcts(grid, tiles, available_tiles, subkey)
        return (grid, available_tiles, key), (valid, action)

    key = jax.random.key(0)
    (final_grid, final_available_tiles, _), (validities, actions) = jax.lax.scan(
        body,
        (grid, available_tiles, key),
        length=3
    )

    def decode_action(action):
        def invalid_action(_):
            return -1, -1   
        tile_idx = action // 81
        position = action % 81
        return jax.lax.cond(
            action >= 0,
            lambda _: (tile_idx, position),
            invalid_action,
            operand=None
        )

    tile_indices, positions = jax.vmap(decode_action)(actions)
    a1, a2, a3 = positions

    return 0, tile_indices, a1, a2, a3

# @partial(jax.jit, static_argnames=["batch_size"])
# def run_mcts_batches(grid, root_tiles, batch_size=100):
#     if batch_size < 1:
#         raise ValueError("batch_size must be positive")

#     batch_size = min(batch_size, NUM_CANDIDATES)
#     num_batches = (MCTS_ITERS + batch_size - 1) // batch_size
#     tree = init_tree()
#     continuations = precompute_continuations(grid, root_tiles)
#     keys = jax.random.split(jax.random.key(0), num_batches)
#     batch_ids = jnp.arange(num_batches)

#     def body(tree, inputs):
#         key, batch_id = inputs
#         active = jnp.arange(batch_size) + batch_id * batch_size < MCTS_ITERS
#         tree = mcts_iteration_batch(
#             tree,
#             continuations,
#             key,
#             active,
#             batch_size,
#         )
#         return tree, None

#     tree, _ = jax.lax.scan(body, tree, (keys, batch_ids))
#     scores = jnp.where(tree["valid"], q_value(tree), -jnp.inf)
#     best_idx = jnp.argmax(scores)
#     tile_idx = best_idx // NUM_ACTIONS
#     action = best_idx % NUM_ACTIONS
#     result_grid, _, _ = step(grid, root_tiles[tile_idx], action)
#     result_grid = clear_lines(result_grid)
#     result_grid = jnp.where(tree["valid"][best_idx], result_grid, grid)

#     return result_grid, tile_idx, action


# @partial(jax.jit, static_argnames=["batch_size"])
# def mcts_iteration_batch(tree, continuations, key, active, batch_size):
#     indices, selected = select_batch(tree, key, batch_size)
#     selected = selected & active
#     tile_indices = indices // NUM_ACTIONS
#     continuation_keys = jax.random.split(key, batch_size)
#     choice_scores = jax.random.uniform(
#         key,
#         (batch_size, continuations["valid"].shape[1]),
#     )
#     choice_scores = jnp.where(
#         continuations["valid"][tile_indices],
#         choice_scores,
#         -jnp.inf,
#     )
#     continuation_indices = jnp.argmax(choice_scores, axis=1)
#     continuation_valid = jnp.any(continuations["valid"][tile_indices], axis=1)
#     selected = selected & continuation_valid

#     selected_grids = continuations["grids"][tile_indices, continuation_indices]
#     values = jax.vmap(rollout)(selected_grids, continuation_keys)
#     values = values + continuations["rewards"][tile_indices, continuation_indices]
#     return update(tree, indices, values, selected)
# # 