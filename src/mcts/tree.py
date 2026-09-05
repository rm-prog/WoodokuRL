import jax
import jax.numpy as jnp

from src.config import GRID_SIZE
from src.env.actions import is_valid_placement_flat
from src.env.tiles import SHIFTED_TILES, SHIFTED_TILES_VALID

NUM_TILES = 3
NUM_ACTIONS = GRID_SIZE * GRID_SIZE
NUM_CANDIDATES = NUM_TILES * NUM_ACTIONS


@jax.jit
def enumerate_candidates():
    """Enumerate candidates as (tile index, placement action)."""
    indices = jnp.arange(NUM_CANDIDATES)
    return indices // NUM_ACTIONS, indices % NUM_ACTIONS


@jax.jit
def init_candidates(grid, tiles):
    tile_indices, actions = enumerate_candidates()
    valid = jax.vmap(is_valid_placement_flat, in_axes=(None, 0, 0))(
        grid,
        tiles[tile_indices],
        actions,
    )

    return {
        "valid": valid,
        "visits": jnp.zeros(NUM_CANDIDATES, dtype=jnp.int32),
        "value_sum": jnp.zeros(NUM_CANDIDATES, dtype=jnp.float32),
    }


@jax.jit
def q_value(tree, indices=None):
    values = tree["value_sum"]
    visits = tree["visits"]
    q = values / (visits + 1e-8)
    return q if indices is None else q[indices]


@jax.jit
def update(tree, indices, values, selected):
    increments = selected.astype(jnp.int32)
    tree["visits"] = tree["visits"].at[indices].add(increments)
    tree["value_sum"] = tree["value_sum"].at[indices].add(
        jnp.where(selected, values, 0.0)
    )
    return tree
