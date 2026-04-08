from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from src.env.tiles import TILES
from src.env.actions import step, has_valid_placements

MAX_STEPS = 10


@partial(jax.jit, static_argnames=["func"])
def simulate_games(num_games, func, key):

    def play_step(carry, _):
        grid, key, score, alive = carry
        key, subkey = jax.random.split(key)
        tiles = generate_tiles(subkey, grid)

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

        (grid, score, alive), _ = lax.scan(
            apply_move,
            (grid, score, alive),
            (tiles_ordered, actions)
        )

        return (grid, key, score, alive), None

    def run_one(key):
        grid = jnp.zeros((9, 9), dtype=int)
        score = 0
        alive = True

        (grid, key, score, alive), _ = lax.scan(
            play_step,
            (grid, key, score, alive),
            None,
            length=MAX_STEPS
        )

        return score

    keys = jax.random.split(key, 100)
    scores = jax.vmap(run_one)(keys)

    return scores

@jax.jit
def generate_tiles(key, field):

    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, len(TILES))
    shuffled_tiles = TILES[perm]

    def check(tile):
        valid, _ = has_valid_placements(field, tile)
        return valid

    valids = jax.vmap(check)(shuffled_tiles)

    has_any = jnp.any(valids)

    def pick(_):
        idx = jnp.argmax(valids.astype(jnp.int32))
        first_tile = shuffled_tiles[idx]

        key2, k1, k2 = jax.random.split(key, 3)
        r1 = jax.random.randint(k1, (), 0, len(TILES))
        r2 = jax.random.randint(k2, (), 0, len(TILES))

        tile2 = TILES[r1]
        tile3 = TILES[r2]

        tiles = jnp.stack([first_tile, tile2, tile3])
        return tiles

    def fallback(_):
        # IMPORTANT: same shape as success case
        key2, k1, k2 = jax.random.split(key, 3)
        r1 = jax.random.randint(k1, (), 0, len(TILES))
        r2 = jax.random.randint(k2, (), 0, len(TILES))
        r3 = jax.random.randint(key2, (), 0, len(TILES))

        return jnp.stack([TILES[r1], TILES[r2], TILES[r3]])

    return jax.lax.cond(
        has_any,
        pick,
        fallback,
        operand=None
    )