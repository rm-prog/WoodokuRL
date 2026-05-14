import jax
import jax.numpy as jnp

from src.beam.beam_search import beam_search_with_order
from src.sim.simulation import generate_tiles
from src.env.actions import step


def apply_sequence(grid, tiles, perm, a1, a2, a3):

    ordered = jnp.take(tiles, perm, axis=0)

    g = grid

    g, _, _ = step(g, ordered[0], a1)
    g, _, _ = step(g, ordered[1], a2)
    g, _, _ = step(g, ordered[2], a3)

    return g


def expand(tree, node, tiles, key, score_fn):

    grid = tree["grid"][node]

    if tiles is None:
        tiles = generate_tiles(key, grid)

    _, perm, a1, a2, a3 = beam_search_with_order(
        grid,
        tiles,
        score_fn,
        beam_width_step1=5,
        beam_width_step2=5
    )

    child_grid = apply_sequence(
        grid,
        tiles,
        perm,
        a1,
        a2,
        a3,
    )

    child_id = tree["next_free"]

    tree = tree.copy()

    tree["grid"] = (
        tree["grid"]
        .at[child_id]
        .set(child_grid)
    )

    children = tree["children"][node]

    slot = jnp.argmax(children == -1)

    tree["children"] = (
        tree["children"]
        .at[node, slot]
        .set(child_id)
    )

    tree["parent"] = (
        tree["parent"]
        .at[child_id]
        .set(node)
    )

    tree["next_free"] = tree["next_free"] + 1

    return tree, child_id, child_grid