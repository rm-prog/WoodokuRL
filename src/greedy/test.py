import jax.numpy as jnp
from functools import partial
from src.greedy.score_func import zero_score
from src.greedy.greedy_agent import greedy_with_order
from src.env.tiles import TILES, print_tiles
from src.env.actions import apply_move_flat
from src.env.game import Game

grid = jnp.zeros((9,9), dtype=int)

tiles = jnp.array([TILES[5], TILES[5], TILES[5]])

greedy_zero = partial(greedy_with_order, score_fn = zero_score)

print_tiles(tiles)
print(greedy_zero(grid, tiles))
tiles = jnp.array([TILES[2], TILES[4], TILES[12]])
print_tiles(tiles)
print(greedy_zero(grid, tiles))

game = Game()