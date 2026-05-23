import jax
import jax.numpy as jnp

from src.mcts.policy import random_policy
from src.beam.beam_search import beam_search_topk
from src.greedy.score_func import empty_lines_score
from src.env.actions import step

MAX_CANDIDATES = 1024
BEAM_K = 10

def init_candidates(grid, root_tiles, key):

    beam = beam_search_topk(
        grid,
        root_tiles,
        score_fn=empty_lines_score,
        K=BEAM_K
    )

    beam_scores, beam_perm, beam_a1, beam_a2, beam_a3 = beam

    beam_grids = jax.vmap(
        lambda p, a1, a2, a3: apply_sequence(grid, root_tiles, p, a1, a2, a3)
    )(beam_perm, beam_a1, beam_a2, beam_a3)

    rand_keys = jax.random.split(key, MAX_CANDIDATES - BEAM_K)

    def make_random(subkey):
        perm, a1, a2, a3, _, valid = random_policy(
            grid,
            root_tiles,
            subkey,
        )

        child_grid = apply_sequence(
            grid,
            root_tiles,
            perm,
            a1,
            a2,
            a3,
        )

        return child_grid, perm, a1, a2, a3, valid

    rand_grids, rand_perm, rand_a1, rand_a2, rand_a3, rand_valid = jax.vmap(make_random)(rand_keys)

    grids = jnp.concatenate([beam_grids, rand_grids], axis=0)
    perms = jnp.concatenate([beam_perm, rand_perm], axis=0)
    a1s   = jnp.concatenate([beam_a1, rand_a1], axis=0)
    a2s   = jnp.concatenate([beam_a2, rand_a2], axis=0)
    a3s   = jnp.concatenate([beam_a3, rand_a3], axis=0)

    valids = jnp.concatenate([
        jnp.ones((BEAM_K,), dtype=bool),
        rand_valid
    ])

    return {
        "grid": grids,
        "perm": perms,
        "a1": a1s,
        "a2": a2s,
        "a3": a3s,
        "valid": valids,
        "visits": jnp.zeros((MAX_CANDIDATES,), dtype=jnp.int32),
        "value_sum": jnp.zeros((MAX_CANDIDATES,), dtype=jnp.float32),
    }

def q_value(tree, i):
    return tree["value_sum"][i] / (tree["visits"][i] + 1e-8)

def update(tree, idx, value):
    tree["visits"] = tree["visits"].at[idx].add(1)
    tree["value_sum"] = tree["value_sum"].at[idx].add(value)
    return tree

def apply_sequence(grid, tiles, perm, a1, a2, a3):

    ordered = jnp.take(tiles, perm, axis=0)

    g = grid

    g, _, _ = step(g, ordered[0], a1)
    g, _, _ = step(g, ordered[1], a2)
    g, _, _ = step(g, ordered[2], a3)

    return g
