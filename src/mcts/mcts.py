import jax
import jax.numpy as jnp

from src.env.actions import apply_move_flat, clear_lines

from src.mcts.tree import (
    init_candidates,
    update,
)
from src.mcts.selection import select
from src.mcts.rollout import rollout

from src.config import (
    MCTS_ITERS, 
    NUM_TILES,
    NUM_ACTIONS,
    NUM_PLACEMENT_TRIPLES,
    PERMUTATIONS
    ) 

@jax.jit
def q_value(tree):
    return tree["value_sum"] / (tree["visits"] + 1e-8)

@jax.jit
def run_mcts(grid, root_tiles):

    tree = init_candidates(grid, root_tiles)

    keys = jax.random.split(jax.random.key(0), MCTS_ITERS)

    def body(tree, key):
        tree = mcts_iteration(tree, key, grid, root_tiles)
        return tree, None

    tree, _ = jax.lax.scan(body, tree, keys)

    q = q_value(tree)
    q = jnp.where(tree["valid"], q, -jnp.inf)
    has_valid = jnp.any(tree["valid"])
    best_idx = jnp.argmax(q)

    perm = best_idx // NUM_PLACEMENT_TRIPLES
    remainder = best_idx % NUM_PLACEMENT_TRIPLES
    p1 = remainder // (NUM_ACTIONS ** 2)
    remainder = remainder % (NUM_ACTIONS ** 2)
    p2 = remainder // NUM_ACTIONS
    p3 = remainder % NUM_ACTIONS
    placements = jnp.take(jnp.array([p1,p2,p3]), PERMUTATIONS[perm], axis=0)

    def apply_best_move(_):
        current_grid = grid
        for i in range(NUM_TILES):
            tile = root_tiles[PERMUTATIONS[perm][i]]
            placement = placements[i]
            current_grid = apply_move_flat(current_grid, tile, placement)
            current_grid = clear_lines(current_grid)
        return current_grid

    current_grid = jax.lax.cond(
        has_valid,
        apply_best_move,
        lambda _: grid,
        operand=None,
    )

    return (
        current_grid,
        PERMUTATIONS[perm],
        placements[0],
        placements[1],
        placements[2]
    )

@jax.jit
def mcts_iteration(tree, key, grid, tiles):

    idx = select(tree)

    def no_valid_move():
        return tree

    def rollout_update():

        permutation_idx = idx // (NUM_ACTIONS ** 3)

        placement_idx = idx % (NUM_ACTIONS ** 3)

        p1 = placement_idx // (NUM_ACTIONS ** 2)

        remainder = placement_idx % (NUM_ACTIONS ** 2)

        p2 = remainder // NUM_ACTIONS

        p3 = remainder % NUM_ACTIONS

        permutation = PERMUTATIONS[permutation_idx]

        [p1, p2, p3] = jnp.take(jnp.array([p1, p2, p3]), permutation, axis=0)

        # Apply the three placements in the selected order
        grid_1 = apply_move_flat(
            grid,
            tiles[permutation[0]],
            p1
        )

        grid_2 = apply_move_flat(
            grid_1,
            tiles[permutation[1]],
            p2
        )

        grid_3 = apply_move_flat(
            grid_2,
            tiles[permutation[2]],
            p3
        )

        value = rollout(
            grid_3,
            key
        )

        return update(
            tree,
            idx,
            value
        )

    return jax.lax.cond(
        idx == -1,
        no_valid_move,
        rollout_update
    )