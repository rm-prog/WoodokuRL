from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from src.env.tiles import TILES
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
        key2, k1, k2 = jax.random.split(key, 3)
        r1 = jax.random.randint(k1, (), 0, len(TILES))
        r2 = jax.random.randint(k2, (), 0, len(TILES))
        r3 = jax.random.randint(key2, (), 0, len(TILES))

        return jnp.stack([TILES[r1], TILES[r2], TILES[r3]])

    return lax.cond(
        has_any,
        pick,
        fallback,
        operand=None
    )