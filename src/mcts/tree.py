import itertools

import jax
import jax.numpy as jnp

from src.config import GRID_SIZE

from src.env.actions import clear_lines
from src.env.tiles import SHIFTED_TILES, SHIFTED_TILES_VALID

NUM_TILES = 3
NUM_ACTIONS = GRID_SIZE * GRID_SIZE
NUM_PERMUTATIONS = 6
NUM_PLACEMENT_TRIPLES = NUM_ACTIONS ** 3

NUM_CANDIDATES = NUM_PLACEMENT_TRIPLES * NUM_PERMUTATIONS

PERMUTATIONS = jnp.array(
    list(itertools.permutations(range(NUM_TILES))),
    dtype=jnp.int32
)


@jax.jit
def enumerate_placements():
    indices = jnp.arange(NUM_PLACEMENT_TRIPLES)

    p1 = indices // (NUM_ACTIONS ** 2)

    remainder = indices % (NUM_ACTIONS ** 2)

    p2 = remainder // NUM_ACTIONS

    p3 = remainder % NUM_ACTIONS

    return jnp.stack(
        [p1, p2, p3],
        axis=1
    )


@jax.jit
def valid_candidates_for_permutation(grid, tiles, permutation):
    """Return valid flat candidates for one tile order."""
    tile_1, tile_2, tile_3 = tiles[permutation]

    shifted_1 = SHIFTED_TILES[tile_1]
    shifted_2 = SHIFTED_TILES[tile_2]
    shifted_3 = SHIFTED_TILES[tile_3]
    shifted_1_flat = shifted_1.reshape(NUM_ACTIONS, GRID_SIZE, GRID_SIZE)
    shifted_2_flat = shifted_2.reshape(NUM_ACTIONS, GRID_SIZE, GRID_SIZE)

    valid_1 = (
        (jnp.einsum("rcij,ij->rc", shifted_1, grid) == 0)
        & SHIFTED_TILES_VALID[tile_1]
    ).reshape(-1)
    grids_1 = jax.vmap(clear_lines)(grid + shifted_1_flat)

    grids_2 = (
        grids_1[:, None, :, :] + shifted_2_flat[None, :, :, :]
    ).reshape(NUM_PLACEMENT_TRIPLES // NUM_ACTIONS, GRID_SIZE, GRID_SIZE)
    valid_2 = (
        (jnp.einsum("rcij,nij->nrc", shifted_2, grids_1) == 0)
        & SHIFTED_TILES_VALID[tile_2]
    ).reshape(-1)
    valid_2 = valid_2 & valid_1.repeat(NUM_ACTIONS)
    grids_2 = jax.vmap(clear_lines)(grids_2)

    valid_3 = (
        (jnp.einsum("rcij,nij->nrc", shifted_3, grids_2) == 0)
        & SHIFTED_TILES_VALID[tile_3]
    ).reshape(-1)
    return valid_3 & valid_2.repeat(NUM_ACTIONS)

@jax.jit
def init_candidates(grid, tiles):
    valid = jax.vmap(
        valid_candidates_for_permutation,
        in_axes=(None, None, 0),
    )(grid, tiles, PERMUTATIONS).reshape(-1)

    return {
        "valid": valid,
        "visits": jnp.zeros(NUM_CANDIDATES, dtype=jnp.int32),
        "value_sum": jnp.zeros(NUM_CANDIDATES, dtype=jnp.float32),
    }

@jax.jit
def q_value(tree, i):
    return tree["value_sum"][i] / (tree["visits"][i] + 1e-8)

@jax.jit
def update(tree, idx, value):
    tree["visits"] = tree["visits"].at[idx].add(1)
    tree["value_sum"] = tree["value_sum"].at[idx].add(value)
    return tree
