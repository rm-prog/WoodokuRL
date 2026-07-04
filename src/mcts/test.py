import jax
import jax.numpy as jnp

from src.mcts.mcts import run_mcts
from src.env.tiles import TILES
from src.sim.simulation import simulate_games

from functools import partial

def make_dummy_grid():
    return jnp.zeros((9, 9), dtype=jnp.int32)


def make_dummy_tiles():
    return jnp.ones((3, 2, 2), dtype=jnp.int32)

# result = run_mcts(make_dummy_grid(), jnp.stack([TILES[0], TILES[5], TILES[5]]))

mcts_one_game = partial(simulate_games, num_games=1, func=run_mcts)
key = jax.random.key(0)
print(mcts_one_game(key))