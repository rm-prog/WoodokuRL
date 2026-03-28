import jax.numpy as jnp
from src.greedy.score_func import zero_score
from src.greedy.greedy_agent import greedy_with_order
from src.env.tiles import TILES, print_tiles

grid = jnp.zeros((9,9), dtype=int)

tiles = jnp.array([TILES[5], TILES[5], TILES[5]])

print_tiles(tiles)
print(greedy_with_order(grid, tiles, zero_score))