import jax
import jax.numpy as jnp
import time

from src.sim.simulation import simulate_games
from src.greedy.greedy_agent import greedy_with_order
from src.greedy.score_func import zero_score

def greedy_zero_fn(grid, tiles):
    return greedy_with_order(grid, tiles, zero_score)


key = jax.random.key(0)

start = time.perf_counter()
scores = simulate_games(100, key, greedy_zero_fn).block_until_ready()
end = time.perf_counter()

print("Time for 100 games:", end - start)
print(scores)
print(jnp.mean(scores))