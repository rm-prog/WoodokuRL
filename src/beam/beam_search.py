import jax
import jax.numpy as jnp
from src.env.actions import step
from src.config import GRID_SIZE
from src.env.tiles import TILE_PERMS

def beam_search_with_order(grid, tiles, score_fn, beam_width=5):
    actions = jnp.arange(GRID_SIZE * GRID_SIZE)

    def solve_for_perm(perm):
        t = jnp.take(tiles, perm, axis=0)
        t1, t2, t3 = t

        # --- Level 1: all 81 placements of t1 ---
        def place1(a1):
            g1, v1, r1 = step(grid, t1, a1)
            score = jnp.where(v1, r1 + score_fn(g1), -jnp.inf)
            return score, g1, r1, v1

        scores1, grids1, rewards1, valids1 = jax.vmap(place1)(actions)
        # scores1: (81,), grids1: (81,9,9)

        # Keep top beam_width by score
        top1_idx = jnp.argsort(scores1)[-beam_width:]          # (K,)
        beam_grids1   = grids1[top1_idx]                        # (K,9,9)
        beam_rewards1 = rewards1[top1_idx]                      # (K,)
        beam_a1s      = top1_idx                                # (K,)

        # --- Level 2: expand K beam nodes over all 81 placements of t2 ---
        def expand2(g1, r1):
            def place2(a2):
                g2, v2, r2 = step(g1, t2, a2)
                score = jnp.where(v2, r1 + r2 + score_fn(g2), -jnp.inf)
                return score, g2, r1 + r2, v2
            return jax.vmap(place2)(actions)  # (81,), (81,9,9), (81,), (81,)

        scores2, grids2, rewards2, valids2 = jax.vmap(expand2)(beam_grids1, beam_rewards1)
        # scores2: (K,81), grids2: (K,81,9,9)

        scores2_flat  = scores2.reshape(-1)                     # (K*81,)
        grids2_flat   = grids2.reshape(-1, GRID_SIZE, GRID_SIZE)# (K*81,9,9)
        rewards2_flat = rewards2.reshape(-1)                    # (K*81,)

        # Keep top beam_width
        top2_idx      = jnp.argsort(scores2_flat)[-beam_width:]# (K,)
        beam_grids2   = grids2_flat[top2_idx]                   # (K,9,9)
        beam_rewards2 = rewards2_flat[top2_idx]                 # (K,)
        # track which a1 and a2 each beam node came from
        beam_a1s2     = beam_a1s[top2_idx // (GRID_SIZE * GRID_SIZE)]  # (K,)
        beam_a2s      = top2_idx % (GRID_SIZE * GRID_SIZE)             # (K,)

        # --- Level 3: expand K beam nodes over all 81 placements of t3 ---
        def expand3(g2, r2):
            def place3(a3):
                g3, v3, r3 = step(g2, t3, a3)
                score = jnp.where(v3, r2 + r3 + score_fn(g3), -jnp.inf)
                return score, a3
            return jax.vmap(place3)(actions)  # (81,), (81,)

        scores3, a3s = jax.vmap(expand3)(beam_grids2, beam_rewards2)
        # scores3: (K,81)

        scores3_flat = scores3.reshape(-1)                      # (K*81,)
        a3s_flat     = a3s.reshape(-1)                          # (K*81,)

        # Pick best overall
        best_idx  = jnp.argmax(scores3_flat)
        best_score = scores3_flat[best_idx]

        beam_node = best_idx // (GRID_SIZE * GRID_SIZE)         # which beam node
        a3_best   = a3s_flat[best_idx]
        a2_best   = beam_a2s[beam_node]
        a1_best   = beam_a1s2[beam_node]

        return best_score, perm, a1_best, a2_best, a3_best

    # Try all 6 permutations — same as greedy_with_order
    results = jax.vmap(solve_for_perm)(TILE_PERMS)
    scores  = results[0]
    idx     = jnp.argmax(scores)
    best    = jax.tree_util.tree_map(lambda x: x[idx], results)
    return best  # (score, perm, a1, a2, a3)