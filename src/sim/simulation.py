from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from src.env.tiles import TILES, SHIFTED_TILES, SHIFTED_TILES_VALID
from src.env.actions import step, has_valid_placements

@partial(jax.jit, static_argnames=["func", "num_games"])
def simulate_games(key, num_games, func):
    def play_step(carry):
        grid, key, score, alive = carry
        key, subkey = jax.random.split(key)
        tiles = generate_tiles(subkey, grid)

        def do_step(_):
            moves = func(grid, tiles)
            _, perm, a1, a2, a3 = moves
            tiles_ordered = jnp.take(tiles, perm, axis=0)
            actions = jnp.array([a1, a2, a3])

            def apply_move(carry, x):
                grid, score, alive = carry
                tile, action = x
                new_grid, valid, reward = step(grid, tile, action)
                score = score + reward * valid
                alive = alive & valid
                grid = jnp.where(valid, new_grid, grid)
                return (grid, score, alive), None

            return lax.scan(
                apply_move,
                (grid, score, alive),
                (tiles_ordered, actions)
            )[0]

        def skip_step(_):
            return grid, score, alive

        grid, score, alive = lax.cond(
            alive,
            do_step,
            skip_step,
            operand=None
        )
        return (grid, key, score, alive)

    def run_one(key):
        grid = jnp.zeros((9, 9), dtype=jnp.int32)
        score = jnp.float32(0)
        alive = jnp.bool_(True)

        def cond_fn(carry):
            _, _, _, alive = carry
            return alive

        def body_fn(carry):
            return play_step(carry)

        grid, key, score, alive = lax.while_loop(
            cond_fn,
            body_fn,
            (grid, key, score, alive)
        )

        return score

    keys = jax.random.split(key, num_games)
    scores = jax.vmap(run_one)(keys)
    return scores

@jax.jit
def all_tiles_valid_mask(field):
    """For every tile (original TILES order), does it have >=1 legal placement
    on `field`? One einsum over all (tile, row, col) at once."""
    overlap = jnp.einsum('trcij,ij->trc', SHIFTED_TILES, field)   # (T, 9, 9)
    legal = (overlap == 0) & SHIFTED_TILES_VALID                   # (T, 9, 9)
    return jnp.any(legal, axis=(1, 2))                              # (T,)


@jax.jit
def generate_tiles(key, field):
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, TILES.shape[0])

    tile_has_valid = all_tiles_valid_mask(field)   # (T,), original tile order
    valids = tile_has_valid[perm]                   # reordered to match shuffle
    has_any = jnp.any(valids)

    def pick(_):
        idx = jnp.argmax(valids.astype(jnp.int32))
        first_idx = perm[idx]
        k1, k2 = jax.random.split(key)
        r1 = jax.random.randint(k1, (), 0, TILES.shape[0])
        r2 = jax.random.randint(k2, (), 0, TILES.shape[0])
        return jnp.stack([first_idx, r1, r2])

    def fallback(_):
        k0, k1, k2 = jax.random.split(key, 3)
        r1 = jax.random.randint(k0, (), 0, TILES.shape[0])
        r2 = jax.random.randint(k1, (), 0, TILES.shape[0])
        r3 = jax.random.randint(k2, (), 0, TILES.shape[0])
        return jnp.stack([r1, r2, r3])

    return lax.cond(has_any, pick, fallback, operand=None)