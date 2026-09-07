import jax
import jax.numpy as jnp

from src.config import MCTS_CONTINUATION_WIDTH
from src.env.actions import clear_lines, step
from src.greedy.score_func import empty_lines_score
from src.mcts.tree import NUM_ACTIONS, NUM_CANDIDATES


@jax.jit
def enumerate_continuations(grid, tiles, root_tile_indices, root_actions):
    root_grids, root_valid, root_rewards = jax.vmap(step)(
        jnp.broadcast_to(grid, (NUM_CANDIDATES, *grid.shape)),
        tiles[root_tile_indices],
        root_actions,
    )
    root_grids = jax.vmap(clear_lines)(root_grids)
    safe_root_grids = jnp.where(
        root_valid[:, None, None],
        root_grids,
        jnp.broadcast_to(grid, root_grids.shape),
    )
    root_rewards = jnp.where(root_valid, root_rewards, 0.0)

    remaining_1 = (root_tile_indices + 1) % 3
    remaining_2 = (root_tile_indices + 2) % 3
    orders = jnp.stack(
        [
            jnp.stack([remaining_1, remaining_2], axis=1),
            jnp.stack([remaining_2, remaining_1], axis=1),
        ],
        axis=1,
    )
    next_tiles = tiles[orders[:, :, 0]]
    final_tiles = tiles[orders[:, :, 1]]

    actions = jnp.arange(NUM_ACTIONS)
    def expand_next(state, tile_choices):
        def place_tile(tile):
            return jax.vmap(step, in_axes=(None, None, 0))(
                state,
                tile,
                actions,
            )

        return jax.vmap(place_tile)(tile_choices)

    next_grids, next_valid, next_rewards = jax.vmap(expand_next)(
        safe_root_grids,
        next_tiles,
    )
    next_grids = jax.vmap(jax.vmap(jax.vmap(clear_lines)))(next_grids)
    next_scores = jnp.where(
        next_valid,
        next_rewards + jax.vmap(jax.vmap(jax.vmap(empty_lines_score)))(next_grids),
        -jnp.inf,
    )
    _, top_indices = jax.lax.top_k(
        next_scores.reshape(NUM_CANDIDATES, -1),
        MCTS_CONTINUATION_WIDTH,
    )

    order_indices = top_indices // NUM_ACTIONS
    next_action_indices = top_indices % NUM_ACTIONS
    candidate_indices = jnp.arange(NUM_CANDIDATES)[:, None]
    selected_next_grids = next_grids[
        candidate_indices,
        order_indices,
        next_action_indices,
    ]
    selected_final_tiles = final_tiles[
        candidate_indices,
        order_indices,
    ]
    selected_next_rewards = next_rewards[
        candidate_indices,
        order_indices,
        next_action_indices,
    ]
    selected_next_valid = next_valid[
        candidate_indices,
        order_indices,
        next_action_indices,
    ]

    def expand_final(states, tile_choices):
        def place_tile(state, tile):
            return jax.vmap(step, in_axes=(None, None, 0))(
                state,
                tile,
                actions,
            )

        return jax.vmap(place_tile)(states, tile_choices)

    final_grids, final_valid, final_rewards = jax.vmap(expand_final)(
        selected_next_grids,
        selected_final_tiles,
    )
    final_grids = jax.vmap(jax.vmap(jax.vmap(clear_lines)))(final_grids)
    final_scores = jnp.where(
        selected_next_valid[:, :, None] & final_valid,
        final_rewards + jax.vmap(jax.vmap(jax.vmap(empty_lines_score)))(final_grids),
        -jnp.inf,
    )
    final_scores = final_scores.reshape(NUM_CANDIDATES, -1)
    _, final_indices = jax.lax.top_k(
        final_scores,
        MCTS_CONTINUATION_WIDTH,
    )

    final_grids = final_grids.reshape(
        NUM_CANDIDATES,
        MCTS_CONTINUATION_WIDTH * NUM_ACTIONS,
        *grid.shape,
    )
    final_valid = final_valid.reshape(
        NUM_CANDIDATES,
        MCTS_CONTINUATION_WIDTH * NUM_ACTIONS,
    )
    final_rewards = final_rewards.reshape(
        NUM_CANDIDATES,
        MCTS_CONTINUATION_WIDTH * NUM_ACTIONS,
    )
    selected_next_rewards = selected_next_rewards.reshape(
        NUM_CANDIDATES,
        MCTS_CONTINUATION_WIDTH,
    )
    selected_next_valid = selected_next_valid.reshape(
        NUM_CANDIDATES,
        MCTS_CONTINUATION_WIDTH,
    )
    cached_grids = final_grids[
        jnp.arange(NUM_CANDIDATES)[:, None],
        final_indices,
    ]
    cached_valid = final_valid[
        jnp.arange(NUM_CANDIDATES)[:, None],
        final_indices,
    ] & selected_next_valid[
        jnp.arange(NUM_CANDIDATES)[:, None],
        final_indices // NUM_ACTIONS,
    ]
    cached_rewards = (
        root_rewards[:, None]
        + selected_next_rewards[
            jnp.arange(NUM_CANDIDATES)[:, None],
            final_indices // NUM_ACTIONS,
        ]
        + final_rewards[
            jnp.arange(NUM_CANDIDATES)[:, None],
            final_indices,
        ]
    )
    cached_scores = final_scores[
        jnp.arange(NUM_CANDIDATES)[:, None],
        final_indices,
    ]
    cached_valid = cached_valid & root_valid[:, None]

    return {
        "grids": cached_grids,
        "valid": cached_valid,
        "rewards": jnp.where(cached_valid, cached_rewards, 0.0),
        "scores": cached_scores,
        "root_valid": root_valid,
    }


@jax.jit
def precompute_continuations(grid, tiles):
    root_tile_indices = jnp.repeat(jnp.arange(3), NUM_ACTIONS)
    root_actions = jnp.tile(jnp.arange(NUM_ACTIONS), 3)
    return enumerate_continuations(grid, tiles, root_tile_indices, root_actions)