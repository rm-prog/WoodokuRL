import jax
import jax.numpy as jnp
from jax import lax

from src.config import GRID_SIZE
from src.env.tiles import TILE_PERMS
from src.env.actions import step, has_valid_placements

def beam_search_with_order(grid, tiles, score_fn, beam_width_step1=81, beam_width_step2=10):

    actions = jnp.arange(GRID_SIZE * GRID_SIZE)

    def solve_for_perm(perm):
        t = jnp.take(tiles, perm, axis=0)
        t1, t2, t3 = t

        def place1(a1):
            g1, v1, r1 = step(grid, t1, a1)

            survives, _ = has_valid_placements(g1, t2)
            s1 = r1 + score_fn(g1)
            score = jnp.where(v1 & survives, s1, -jnp.inf)

            return score, g1, r1, v1

        scores1, grids1, rewards1, valids1 = jax.vmap(place1)(actions)

        _, top1_idx = lax.top_k(scores1, beam_width_step1)

        beam_grids1   = grids1[top1_idx]
        beam_rewards1 = rewards1[top1_idx]
        beam_valids1  = valids1[top1_idx]
        beam_a1s      = top1_idx

        def expand2(g1, r1, v1):
            def place2(a2):
                g2, v2, r2 = step(g1, t2, a2)

                survives, _ = has_valid_placements(g2, t3)
                v12 = v1 & v2 & survives
                s2 = r1 + r2 + score_fn(g2)

                score = jnp.where(v12, s2, -jnp.inf)

                return score, g2, r1 + r2, v12

            return jax.vmap(place2)(actions)

        scores2, grids2, rewards2, valids2 = jax.vmap(expand2)(
            beam_grids1, beam_rewards1, beam_valids1
        )

        K = beam_width_step2
        scores2_flat  = scores2.reshape(-1)
        grids2_flat   = grids2.reshape(-1, GRID_SIZE, GRID_SIZE)
        rewards2_flat = rewards2.reshape(-1)
        valids2_flat  = valids2.reshape(-1)

        _, top2_idx = lax.top_k(scores2_flat, beam_width_step2)

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

                score = jnp.where(v123, s3, -jnp.inf)

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

def beam_search_topk(
    grid,
    tiles,
    score_fn,
    K=10,
    beam_width_step1=81,
    beam_width_step2=10,
):

    actions = jnp.arange(GRID_SIZE * GRID_SIZE)

    def solve_for_perm(perm):

        t = jnp.take(tiles, perm, axis=0)
        t1, t2, t3 = t

        # ---------- step 1 ----------

        def place1(a1):

            g1, v1, r1 = step(grid, t1, a1)

            survives, _ = has_valid_placements(g1, t2)

            score = jnp.where(
                v1 & survives,
                r1 + score_fn(g1),
                -jnp.inf,
            )

            return score, g1, r1, v1

        scores1, grids1, rewards1, valids1 = jax.vmap(place1)(actions)

        _, top1 = lax.top_k(scores1, beam_width_step1)

        beam_g1 = grids1[top1]
        beam_r1 = rewards1[top1]
        beam_v1 = valids1[top1]
        beam_a1 = top1

        # ---------- step 2 ----------

        def expand2(g1, r1, v1):

            def place2(a2):

                g2, v2, r2 = step(g1, t2, a2)

                survives, _ = has_valid_placements(g2, t3)

                valid = v1 & v2 & survives

                score = jnp.where(
                    valid,
                    r1 + r2 + score_fn(g2),
                    -jnp.inf,
                )

                return score, g2, r1 + r2, valid

            return jax.vmap(place2)(actions)

        scores2, grids2, rewards2, valids2 = jax.vmap(expand2)(
            beam_g1,
            beam_r1,
            beam_v1,
        )

        scores2 = scores2.reshape(-1)
        grids2 = grids2.reshape(-1, GRID_SIZE, GRID_SIZE)
        rewards2 = rewards2.reshape(-1)
        valids2 = valids2.reshape(-1)

        parent_idx = jnp.repeat(
            jnp.arange(beam_width_step1),
            GRID_SIZE * GRID_SIZE,
        )

        parent_a1 = beam_a1[parent_idx]
        parent_a2 = jnp.tile(actions, beam_width_step1)

        _, top2 = lax.top_k(scores2, beam_width_step2)

        beam_g2 = grids2[top2]
        beam_r2 = rewards2[top2]
        beam_v2 = valids2[top2]

        beam_a1 = parent_a1[top2]
        beam_a2 = parent_a2[top2]

        # ---------- step 3 ----------

        def expand3(g2, r2, v2):

            def place3(a3):

                g3, v3, r3 = step(g2, t3, a3)

                valid = v2 & v3

                score = jnp.where(
                    valid,
                    r2 + r3 + score_fn(g3),
                    -jnp.inf,
                )

                return score, a3, valid

            return jax.vmap(place3)(actions)

        scores3, a3s, valids3 = jax.vmap(expand3)(
            beam_g2,
            beam_r2,
            beam_v2,
        )

        scores3 = scores3.reshape(-1)
        a3s = a3s.reshape(-1)
        valids3 = valids3.reshape(-1)

        # repeat parents so every a3 has its own (a1,a2)

        final_a1 = jnp.repeat(beam_a1, GRID_SIZE * GRID_SIZE)
        final_a2 = jnp.repeat(beam_a2, GRID_SIZE * GRID_SIZE)

        final_perm = jnp.repeat(
            perm[None, :],
            scores3.shape[0],
            axis=0,
        )

        return (
            scores3,
            final_perm,
            final_a1,
            final_a2,
            a3s,
            valids3,
        )

    results = jax.vmap(solve_for_perm)(TILE_PERMS)

    scores = results[0].reshape(-1)
    perms = results[1].reshape(-1, 3)
    a1s = results[2].reshape(-1)
    a2s = results[3].reshape(-1)
    a3s = results[4].reshape(-1)
    valids = results[5].reshape(-1)

    scores = jnp.where(valids, scores, -jnp.inf)

    _, top_idx = lax.top_k(scores, K)

    return (
        scores[top_idx],
        perms[top_idx],
        a1s[top_idx],
        a2s[top_idx],
        a3s[top_idx],
        valids[top_idx],
    )