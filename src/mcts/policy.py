import jax
import jax.numpy as jnp
from jax import lax

from src.beam.beam_search import beam_search_with_order
from src.env.actions import step, has_valid_placements
from src.greedy.score_func import empty_lines_score

def greedy_policy(grid, tiles, score_fn=empty_lines_score, beam_width_step1=5, beam_width_step2=5):

    score, perm, a1, a2, a3 = beam_search_with_order(
        grid,
        tiles,
        score_fn,
        beam_width_step1,
        beam_width_step2
    )

    g, v1, _ = step(grid, tiles[perm[0]], a1)
    g, v2, _ = step(g, tiles[perm[1]], a2)
    g, v3, _ = step(g, tiles[perm[2]], a3)

    valid = v1 & v2 & v3

    return perm, a1, a2, a3, valid

def random_policy(grid, tiles, key):

    def sample_one(g, tile, key):

        has_any, valid_actions = has_valid_placements(g, tile)

        def no_valid(_):
            return 0, key, False

        def has_valid_branch(_):

            new_key, subkey = jax.random.split(key)

            # shuffle full action space
            perm = jax.random.permutation(subkey, 81)

            shuffled = valid_actions[perm]

            # find first valid action (not -1)
            idx = jnp.argmax(shuffled != -1)

            action = shuffled[idx]

            return action, new_key, True

        return lax.cond(
            has_any,
            has_valid_branch,
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
        perm, a1, a2, a3, valid = greedy_policy(grid, tiles, score_fn)
        return perm, a1, a2, a3, key, valid

    def use_random_branch(_):
        return random_policy(grid, tiles, key)

    return lax.cond(
        use_random,
        use_random_branch,
        use_greedy,
        operand=None
    )