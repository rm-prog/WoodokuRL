import jax
import jax.numpy as jnp
from jax import lax

from src.sim.simulation import generate_tiles
from src.env.actions import step
from src.mcts.policy import random_policy


def apply_sequence(grid, tiles, perm, a1, a2, a3):

    ordered = jnp.take(tiles, perm, axis=0)

    g = grid

    g, _, _ = step(g, ordered[0], a1)
    g, _, _ = step(g, ordered[1], a2)
    g, _, _ = step(g, ordered[2], a3)

    return g


def expand(tree, node, tiles, key):

    grid = tree["grid"][node]

    if tiles is None:
        key, subkey = jax.random.split(key)
        tiles = generate_tiles(subkey, grid)

    perm, a1, a2, a3, key, valid = random_policy(
        grid,
        tiles,
        key,
    )

    def invalid_branch(_):
        return tree, -1, grid
    
    def valid_branch(_):

        child_grid = apply_sequence(
            grid,
            tiles,
            perm,
            a1,
            a2,
            a3,
        )

        child_id = tree["next_free"]

        new_tree = tree.copy()

        new_tree["grid"] = (
            new_tree["grid"]
            .at[child_id]
            .set(child_grid)
        )

        children = new_tree["children"][node]

        slot = jnp.argmax(children == -1)

        new_tree["children"] = (
            new_tree["children"]
            .at[node, slot]
            .set(child_id)
        )

        new_tree["parent"] = new_tree["parent"].at[child_id].set(node)

        new_tree["next_free"] = new_tree["next_free"] + 1

        return new_tree, child_id, child_grid

    return lax.cond(
        valid,
        valid_branch,
        invalid_branch,
        operand=None
    )