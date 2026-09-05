import itertools

import jax
import jax.numpy as jnp

from src.config import GRID_SIZE

from src.env.actions import has_valid_placements, apply_move_flat_valid, apply_move_flat, step

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
def valid_placements(grid, tile):
    """
    Generate all possible placements for one tile.

    Returns:
        placement_valid: (NUM_ACTIONS,)
        grids:           (NUM_ACTIONS, GRID_SIZE, GRID_SIZE)
    """

    actions = jnp.arange(NUM_ACTIONS)

    grids, _, _ = jax.vmap(
        step,
        in_axes=(None, None, 0)
    )(grid, tile, actions)

    _, placement_valid = has_valid_placements(
        grid,
        tile
    )

    return placement_valid, grids

@jax.jit
def check_placement_triple(grid, tiles, placements):
    """
    Check all 6 permutations of one placement triple.

    Args:
        grid:
            (GRID_SIZE, GRID_SIZE)

        tiles:
            (3, ...)

        placements:
            (3,)

    Returns:
        canonical:
            (6,)

        permutation_grids:
            (6, GRID_SIZE, GRID_SIZE)
    """

    def simulate(permutation):

        current_grid = grid
        sequence_valid = True

        permuted_placements = jnp.take(placements, permutation, axis=0)

        for i in range(NUM_TILES):

            tile = tiles[permutation[i]]
            placement = permuted_placements[i]

            current_grid, placement_valid, _ = step(
                current_grid,
                tile,
                placement
            )

            sequence_valid &= placement_valid

        return sequence_valid, current_grid

    permutation_valid, permutation_grids = jax.vmap(
        simulate
    )(PERMUTATIONS)

    # ---------------------------------------------------------
    # Compare final grids
    # ---------------------------------------------------------

    equal_grids = jax.vmap(
        lambda g: jnp.all(
            permutation_grids == g,
            axis=(1, 2)
        )
    )(permutation_grids)

    # equal_grids[i, j] = True if permutations i and j
    # result in the same final grid.

    # ---------------------------------------------------------
    # Remove redundant permutations
    # ---------------------------------------------------------

    # Only look at permutations BEFORE the current one.
    earlier_equal = jnp.tril(
        equal_grids,
        k=-1
    )

    duplicate = jnp.any(
        earlier_equal,
        axis=1
    )

    canonical = (
        permutation_valid
        & ~duplicate
    )

    return canonical

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
def init_candidates(grid, tiles):

    placements = enumerate_placements()

    valid_by_placement = jax.vmap(
        check_placement_triple,
        in_axes=(None, None, 0)
        )(
            grid,
            tiles,
            placements
        )

    valid = valid_by_placement.T.reshape(-1)

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
