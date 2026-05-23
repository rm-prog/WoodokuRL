import jax
import jax.numpy as jnp

from src.mcts.mcts import run_mcts
from src.env.tiles import TILES

def make_dummy_grid():
    return jnp.zeros((9, 9), dtype=jnp.int32)


def make_dummy_tiles():
    return jnp.ones((3, 2, 2), dtype=jnp.int32)

result = run_mcts(make_dummy_grid(), jnp.stack([TILES[0], TILES[5], TILES[5]]))

print(result["visits"])
print(result["value_sum"])