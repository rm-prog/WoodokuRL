from functools import partial

import jax
import jax.numpy as jnp

from src.config import MCTS_ITERS
from src.env.actions import clear_lines, step
from src.mcts.rollout import rollout
from src.mcts.selection import select
from src.mcts.expansion import expand
from src.mcts.tree import init_tree, update, update_node, add_child
from src.mcts.backprop import backprop

from src.env.tiles import TILE_PERMS

@jax.jit
def q_value(tree):
    return tree["value_sum"] / (tree["visits"] + 1e-8)

@jax.jit
def run_mcts(grid, root_tiles, tiles_available, key):
    tree = init_tree()
    root = 0

    tree = add_child(tree, -1, -1, 0, root, tiles_available)

    keys = jax.random.split(key, MCTS_ITERS)

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
    idx, selected_grid, total_reward = select(tree, grid, root_tiles)
    tree, chosen_child, chosen_grid, reward = expand(tree, idx, selected_grid, root_tiles, key)
    total_reward = total_reward + reward
    available_tiles = tree["tiles_available"][chosen_child]
    value = rollout(chosen_grid, root_tiles, available_tiles, key, total_reward)
    tree = backprop(tree, chosen_child, value)

    return tree

@jax.jit
def three_move_mcts(grid, tiles):

    available_tiles = jnp.ones(3, dtype=jnp.bool_)

    def one_mcts(grid, tiles, available_tiles, key):
        action = run_mcts(grid, tiles, available_tiles, key)

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
