import jax
import jax.numpy as jnp
from functools import partial
from src.env.tiles import TILES, TILE_PERMS
from src.config import GRID_SIZE
from src.env.actions import step

from functools import partial
import jax
import jax.numpy as jnp

BIG = 1000.0


@partial(jax.jit, static_argnames=["score_fn"])
def greedy_with_order(grid, tiles, score_fn):

    actions = jnp.arange(GRID_SIZE * GRID_SIZE)

    def solve_for_perm(perm):
        t = jnp.take(tiles, perm, axis=0)
        t1, t2, t3 = t

        def step1(a1):
            g1, v1, r1 = step(grid, t1, a1)

            def step2(a2):
                g2, v2, r2 = step(g1, t2, a2)

                def step3(a3):
                    g3, v3, r3 = step(g2, t3, a3)

                    v12 = v1 & v2
                    v123 = v12 & v3

                    s1 = r1 + score_fn(g1)
                    s2 = r1 + r2 + score_fn(g2)
                    s3 = r1 + r2 + r3 + score_fn(g3)

                    p1 = jnp.where(v1, 1 * BIG + s1, -jnp.inf)
                    p2 = jnp.where(v12, 2 * BIG + s2, -jnp.inf)
                    p3 = jnp.where(v123, 3 * BIG + s3, -jnp.inf)

                    best = jnp.maximum(jnp.maximum(p1, p2), p3)

                    return best, a3

                scores3, a3s = jax.vmap(step3)(actions)
                idx3 = jnp.argmax(scores3)

                return scores3[idx3], a2, a3s[idx3]

            scores2, a2s, a3s = jax.vmap(step2)(actions)
            idx2 = jnp.argmax(scores2)

            return scores2[idx2], a1, a2s[idx2], a3s[idx2]

        scores1, a1s, a2s, a3s = jax.vmap(step1)(actions)
        idx1 = jnp.argmax(scores1)

        return (
            scores1[idx1],
            perm,
            a1s[idx1],
            a2s[idx1],
            a3s[idx1],
        )

    results = jax.vmap(solve_for_perm)(TILE_PERMS)

    scores = results[0]
    idx = jnp.argmax(scores)

    best = jax.tree_util.tree_map(lambda x: x[idx], results)

    return best