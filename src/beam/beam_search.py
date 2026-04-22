import jax
import jax.numpy as jnp
from jax import lax

from src.config import GRID_SIZE
from src.env.tiles import TILE_PERMS
from src.env.actions import step

BIG = 1000.0

def beam_search_with_order(grid, tiles, score_fn, beam_width=10):

    actions = jnp.arange(GRID_SIZE * GRID_SIZE)

    def solve_for_perm(perm):
        t = jnp.take(tiles, perm, axis=0)
        t1, t2, t3 = t

        def place1(a1):
            g1, v1, r1 = step(grid, t1, a1)

            s1 = r1 + score_fn(g1)
            score = jnp.where(v1, 1 * BIG + s1, -jnp.inf)

            return score, g1, r1, v1

        scores1, grids1, rewards1, valids1 = jax.vmap(place1)(actions)

        _, top1_idx = lax.top_k(scores1, beam_width)

        beam_grids1   = grids1[top1_idx]
        beam_rewards1 = rewards1[top1_idx]
        beam_valids1  = valids1[top1_idx]
        beam_a1s      = top1_idx

        def expand2(g1, r1, v1):
            def place2(a2):
                g2, v2, r2 = step(g1, t2, a2)

                v12 = v1 & v2
                s2 = r1 + r2 + score_fn(g2)

                score = jnp.where(v12, 2 * BIG + s2, -jnp.inf)

                return score, g2, r1 + r2, v12

            return jax.vmap(place2)(actions)

        scores2, grids2, rewards2, valids2 = jax.vmap(expand2)(
            beam_grids1, beam_rewards1, beam_valids1
        )

        K = beam_width
        scores2_flat  = scores2.reshape(-1)
        grids2_flat   = grids2.reshape(-1, GRID_SIZE, GRID_SIZE)
        rewards2_flat = rewards2.reshape(-1)
        valids2_flat  = valids2.reshape(-1)

        _, top2_idx = lax.top_k(scores2_flat, beam_width)

        beam_grids2   = grids2_flat[top2_idx]
        beam_rewards2 = rewards2_flat[top2_idx]
        beam_valids2  = valids2_flat[top2_idx]

        parent_idx = top2_idx // (GRID_SIZE * GRID_SIZE)
        beam_a1s2  = beam_a1s[parent_idx]
        beam_a2s   = top2_idx % (GRID_SIZE * GRID_SIZE)

        def expand3(g2, r2, v2):
            def place3(a3):
                g3, v3, r3 = step(g2, t3, a3)

                v123 = v2 & v3
                s3 = r2 + r3 + score_fn(g3)

                score = jnp.where(v123, 3 * BIG + s3, -jnp.inf)

                return score, a3

            return jax.vmap(place3)(actions)

        scores3, a3s = jax.vmap(expand3)(
            beam_grids2, beam_rewards2, beam_valids2
        )

        scores3_flat = scores3.reshape(-1)
        a3s_flat     = a3s.reshape(-1)

        best_idx   = jnp.argmax(scores3_flat)
        best_score = scores3_flat[best_idx]

        beam_node = best_idx // (GRID_SIZE * GRID_SIZE)

        a1_best = beam_a1s2[beam_node]
        a2_best = beam_a2s[beam_node]
        a3_best = a3s_flat[best_idx]

        return best_score, perm, a1_best, a2_best, a3_best

    results = jax.vmap(solve_for_perm)(TILE_PERMS)

    scores = results[0]
    idx = jnp.argmax(scores)

    best = jax.tree_util.tree_map(lambda x: x[idx], results)

    return best