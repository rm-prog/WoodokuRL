import jax
import jax.numpy as jnp
from jax import lax

from src.env.actions import step
from src.sim.simulation import generate_tiles

@jax.jit
def rollout(grid, root_tiles, available_tiles, key, depth=5):

    def rollout_from_available(tiles, available_tiles, key):
        available_indices = jnp.nonzero(available_tiles, size=3, fill_value=0)[0]
        num_available = jnp.sum(available_tiles)
        key, subkey = jax.random.split(key)
        random_index = jax.random.randint(subkey, (), 0, num_available)
        chosen_tile_index = available_indices[random_index]
        action = jax.random.randint(subkey, (), 0, 81)
        new_grid, _, reward = step(grid, tiles[chosen_tile_index], action)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        available_tiles = available_tiles.at[chosen_tile_index].set(False)
        return new_grid, available_tiles, reward, key

    def rollout_from_random(key):
        key, subkey = jax.random.split(key)
        tiles = generate_tiles(subkey, grid)
        available_tiles = jnp.ones(3, dtype=jnp.bool_)
        return rollout_from_available(tiles, available_tiles, key)

    def body(carry, _):
        grid, available_tiles, total_reward, key = carry

        def valid_path(_):
            new_grid, new_available_tiles, reward, new_key = rollout_from_available(
                root_tiles,
                available_tiles,
                key,
            )
            reward = jnp.asarray(reward, dtype=jnp.float32)
            return (new_grid, new_available_tiles, total_reward + reward, new_key)

        def invalid_path(_):
            return rollout_from_random(key)

        grid, available_tiles, total_reward, key = lax.cond(
            jnp.any(available_tiles),
            valid_path,
            invalid_path,
            operand=None
        )

        return (grid, available_tiles, total_reward, key), None

    (final_grid, _, total_reward, _), _ = lax.scan(
        body,
        (grid, available_tiles, 0.0, key),
        xs=None,
        length=depth,
    )

    return total_reward
