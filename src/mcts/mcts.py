from functools import partial

import jax
import jax.numpy as jnp

from src.config import MCTS_CONTINUATION_WIDTH, MCTS_ITERS
from src.env.actions import clear_lines, step
from src.greedy.score_func import empty_lines_score
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
    keys = jax.random.split(jax.random.key(0), num_batches)
    batch_ids = jnp.arange(num_batches)

    def body(tree, inputs):
        key, batch_id = inputs
        active = jnp.arange(batch_size) + batch_id * batch_size < MCTS_ITERS
        tree = mcts_iteration_batch(tree, key, grid, root_tiles, active, batch_size)
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
def mcts_iteration_batch(tree, key, grid, tiles, active, batch_size):
    indices, selected = select_batch(tree, key, batch_size)
    selected = selected & active
    tile_indices = indices // NUM_ACTIONS
    actions = indices % NUM_ACTIONS

    candidate_grids, valid, immediate_rewards = jax.vmap(step)(
        jnp.broadcast_to(grid, (batch_size, *grid.shape)),
        tiles[tile_indices],
        actions,
    )
    candidate_grids = jax.vmap(clear_lines)(candidate_grids)
    continuation_keys = jax.random.split(key, batch_size)
    values = jax.vmap(evaluate_continuations, in_axes=(0, None, 0, 0, 0))(
        candidate_grids,
        tiles,
        tile_indices,
        immediate_rewards,
        continuation_keys,
    )
    selected = selected & valid
    return update(tree, indices, values, selected)


@jax.jit
def evaluate_continuations(grid, tiles, first_tile_idx, first_reward, key):
    remaining = jnp.array(
        [(first_tile_idx + 1) % 3, (first_tile_idx + 2) % 3],
        dtype=jnp.int32,
    )
    orders = jnp.array([[remaining[0], remaining[1]], [remaining[1], remaining[0]]])
    first_tile = tiles[orders[:, 0]]
    second_tile = tiles[orders[:, 1]]

    actions = jnp.arange(NUM_ACTIONS)
    first_grids, first_valid, first_rewards = jax.vmap(
        lambda tile: jax.vmap(step, in_axes=(None, None, 0))(
            grid, tile, actions
        )
    )(first_tile)
    first_grids = jax.vmap(jax.vmap(clear_lines))(first_grids)
    first_scores = jnp.where(
        first_valid,
        first_rewards + jax.vmap(jax.vmap(empty_lines_score))(first_grids),
        -jnp.inf,
    )
    _, top_indices = jax.lax.top_k(first_scores.reshape(-1), MCTS_CONTINUATION_WIDTH)

    order_indices = top_indices // NUM_ACTIONS
    placement_indices = top_indices % NUM_ACTIONS
    selected_grids = first_grids[order_indices, placement_indices]
    selected_tiles = second_tile[order_indices]

    final_grids, final_valid, final_rewards = jax.vmap(
        lambda state, tile: jax.vmap(step, in_axes=(None, None, 0))(
            state, tile, actions
        )
    )(selected_grids, selected_tiles)
    final_grids = jax.vmap(jax.vmap(clear_lines))(final_grids)
    final_scores = jnp.where(
        final_valid,
        final_rewards
        + jax.vmap(jax.vmap(empty_lines_score))(final_grids),
        -jnp.inf,
    )
    flat_scores = final_scores.reshape(-1)
    best = jnp.argmax(flat_scores)
    best_grid = final_grids.reshape(-1, 9, 9)[best]
    continuation_valid = jnp.isfinite(flat_scores[best])
    future_value = jax.lax.cond(
        continuation_valid,
        lambda _: rollout(best_grid, key),
        lambda _: 0.0,
        operand=None,
    )
    continuation_score = jnp.where(continuation_valid, flat_scores[best], 0.0)
    return first_reward + continuation_score + future_value
