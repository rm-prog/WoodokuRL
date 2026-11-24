import jax
import jax.numpy as jnp
from jax import lax

from src.env.actions import step
from src.sim.simulation import generate_tiles
from src.env.tiles import SHIFTED_TILES, SHIFTED_TILES_VALID

@jax.jit
def rollout(grid, root_tiles, available_tiles, key, previous_reward, depth=5):

    def rollout_from_available(current_grid, tiles, available_tiles, key):
        next_key, sample_key = jax.random.split(key)
        tile_slot, position, has_legal_move = sample_legal_move(
            current_grid, tiles, available_tiles, sample_key
        )

        def no_legal_move(_):
            return (
                current_grid,
                available_tiles,
                jnp.float32(0),
                next_key,
                jnp.bool_(False),
            )

        def legal_move(_):
            new_grid, valid, reward = step(
                current_grid, tiles[tile_slot], position
            )
            new_available_tiles = available_tiles.at[tile_slot].set(False)
            return (
                jnp.where(valid, new_grid, current_grid),
                new_available_tiles,
                jnp.where(valid, reward, 0).astype(jnp.float32),
                next_key,
                valid,
            )

        return lax.cond(
            has_legal_move, legal_move, no_legal_move, operand=None
        )

    def rollout_from_random(current_grid, key):
        next_key, tiles_key = jax.random.split(key)
        tiles = generate_tiles(tiles_key, current_grid)
        available = jnp.ones(3, dtype=jnp.bool_)
        return rollout_from_available(current_grid, tiles, available, next_key)

    def body(carry, _):
        grid, available, total_reward, key, active = carry

        def do_move(_):
            def use_root_tiles(_):
                return rollout_from_available(grid, root_tiles, available, key)

            def draw_new_tiles(_):
                return rollout_from_random(grid, key)

            new_grid, new_available, reward, new_key, has_move = lax.cond(
                jnp.any(available),
                use_root_tiles,
                draw_new_tiles,
                operand=None,
            )
            return (
                new_grid,
                new_available,
                total_reward + reward,
                new_key,
                has_move,
            )

        def stay_inactive(_):
            return carry

        new_carry = lax.cond(active, do_move, stay_inactive, operand=None)
        return new_carry, None

    (final_grid, _, total_reward, _, _), _ = lax.scan(
        body,
        (grid, available_tiles, 0.0, key, jnp.bool_(True)),
        xs=None,
        length=depth,
    )

    return total_reward + previous_reward


def sample_legal_move(grid, tiles, available_tiles, key):
    # tiles: three tile IDs; available_tiles: bool mask of shape (3,)
    shifted = SHIFTED_TILES[tiles]  # (3, 9, 9, 9, 9)

    overlaps = jnp.einsum(
        "trcij,ij->trc",
        shifted,
        grid,
    )  # (3, 9, 9)

    legal = (
        (overlaps == 0)
        & SHIFTED_TILES_VALID[tiles]
        & available_tiles[:, None, None]
    ).reshape(-1)  # (3 * 81,)

    legal_count = jnp.sum(legal)
    legal_indices = jnp.nonzero(legal, size=243, fill_value=0)[0]

    def sample(_):
        choice = jax.random.randint(key, (), 0, legal_count)
        flat_index = legal_indices[choice]
        tile_slot = flat_index // 81
        position = flat_index % 81
        return tile_slot, position, jnp.bool_(True)

    def no_legal_move(_):
        return (
            jnp.int32(-1),
            jnp.int32(-1),
            jnp.bool_(False),
        )

    return lax.cond(legal_count > 0, sample, no_legal_move, operand=None)
