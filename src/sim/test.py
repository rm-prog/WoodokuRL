import jax
import jax.numpy as jnp
import time
from functools import partial

from src.sim.simulation import simulate_games
from src.greedy.greedy_agent import greedy_with_order
from src.greedy.score_func import zero_score, empty_lines_score

greedy_zero_fn = partial(greedy_with_order, score_fn=zero_score)
greedy_empty_lines_fn = partial(greedy_with_order, score_fn=empty_lines_score)

N = 5
master_key = jax.random.key(0)
keys = jax.random.split(master_key, N)

greedy_5_games = partial(simulate_games, num_games=5, func=greedy_empty_lines_fn)

sum_of_averages = 0
total_time = 0

for k in keys:
    start = time.perf_counter()
    s = greedy_5_games(k).block_until_ready()
    elapsed = time.perf_counter() - start
    total_time = total_time + elapsed
    print(f"Time: {elapsed:.2f}s")
    print(s)
    sum_of_averages = sum_of_averages + jnp.average(s)

print("Average: ", sum_of_averages / N)
print("Total time: ", total_time)