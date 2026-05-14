import jax
import jax.numpy as jnp
from jax import lax

from src.beam.beam_search import beam_search_with_order
from src.env.actions import step, has_valid_placements
from src.greedy.score_func import empty_lines_score

def greedy_policy(grid, tiles, score_fn=empty_lines_score, beam_width_step1=5, beam_width_step2=5):

    return beam_search_with_order(
        grid,
        tiles,
        score_fn,
        beam_width_step1,
        beam_width_step2
    )

def random_policy(grid, tiles, key):

    def sample_one(g, tile, key):

        valid, _ = has_valid_placements(g, tile)
        valid_idx = jnp.where(valid)[0]

        def no_valid(_):
            return 0, key, False

        def has_valid(_):
            key, subkey = jax.random.split(key)

            idx = jax.random.randint(
                subkey,
                (),
                0,
                valid_idx.shape[0],
            )

            return valid_idx[idx], key, True

        return lax.cond(
            jnp.any(valid),
            has_valid,
            no_valid,
            operand=None
        )

    key1, key2, key3, perm_key = jax.random.split(key, 4)

    perm = jax.random.permutation(perm_key, 3)

    t0 = tiles[perm[0]]
    t1 = tiles[perm[1]]
    t2 = tiles[perm[2]]

    a1, key1, v1 = sample_one(grid, t0, key1)
    g1, _, _ = step(grid, t0, a1)

    a2, key2, v2 = sample_one(g1, t1, key2)
    g2, _, _ = step(g1, t1, a2)

    a3, key3, v3 = sample_one(g2, t2, key3)

    valid = v1 & v2 & v3

    return perm, a1, a2, a3, key3, valid

def hybrid_policy(grid, tiles, key, score_fn=empty_lines_score, eps=0.1):

    key, subkey = jax.random.split(key)
    use_random = jax.random.uniform(subkey) < eps

    def use_greedy(_):
        return greedy_policy(grid, tiles, score_fn)

    def use_random_branch(_):
        return random_policy(grid, tiles, key)

    return lax.cond(
        use_random,
        use_random_branch,
        use_greedy,
        operand=None
    )