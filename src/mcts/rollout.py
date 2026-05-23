import jax
import jax.numpy as jnp
from jax import lax

from src.env.actions import step
from src.sim.simulation import generate_tiles
from src.mcts.policy import hybrid_policy


def rollout(grid, key, depth=10, eps=0.2):

    def body(carry, _):
        grid, key, total_reward = carry

        key, subkey = jax.random.split(key)
        tiles = generate_tiles(subkey, grid)

        perm, a1, a2, a3, key, valid = hybrid_policy(
            grid,
            tiles,
            key,
            eps=eps,
        )

        ordered = jnp.take(tiles, perm, axis=0)

        def valid_path(_):
            g1, _, r1 = step(grid, ordered[0], a1)
            g2, _, r2 = step(g1, ordered[1], a2)
            g3, _, r3 = step(g2, ordered[2], a3)

            reward = r1 + r2 + r3
            return (g3, key, total_reward + reward)

        def invalid_path(_):
            return (grid, key, total_reward)

        grid, key, total_reward = lax.cond(
            valid,
            valid_path,
            invalid_path,
            operand=None
        )

        return (grid, key, total_reward), None

    (final_grid, _, total_reward), _ = lax.scan(
        body,
        (grid, key, 0.0),
        xs=None,
        length=depth,
    )

    return total_reward