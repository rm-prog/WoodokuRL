import jax
import jax.numpy as jnp
from functools import partial
import time

from src.sim.simulation import simulate_games
from src.greedy.score_func import empty_lines_score, zero_score
from src.beam.beam_search import beam_search_with_order

beam_empty_lines = partial(beam_search_with_order, score_fn=empty_lines_score)

N = 5
master_key = jax.random.key(0)
keys = jax.random.split(master_key, N)

beam_5_games = partial(simulate_games, num_games=10, func=beam_empty_lines)

sum_of_averages = 0
total_time = 0

for k in keys:
    start = time.perf_counter()
    s = beam_5_games(k).block_until_ready()
    elapsed = time.perf_counter() - start
    total_time = total_time + elapsed
    print(f"Time: {elapsed:.2f}s")
    print(s)
    sum_of_averages = sum_of_averages + jnp.average(s)

print("Average: ", sum_of_averages / N)
print("Total time: ", total_time)