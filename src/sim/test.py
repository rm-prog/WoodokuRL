import jax
import jax.numpy as jnp
import time
from functools import partial

from src.sim.simulation import simulate_games
from src.greedy.greedy_agent import greedy_with_order
from src.greedy.score_func import zero_score, empty_lines_score

greedy_zero_fn = partial(greedy_with_order, score_fn=zero_score)
greedy_empty_lines_fn = partial(greedy_with_order, score_fn=empty_lines_score)

key = jax.random.key(4)
key1 = jax.random.key(0)
key2 = jax.random.key(2)
key3 = jax.random.key(3)
key4 = jax.random.key(1)

greedy_5_games = partial(simulate_games, num_games=5, func=greedy_empty_lines_fn)

for k in [key]:
    start = time.perf_counter()
    s = greedy_5_games(k).block_until_ready()
    print(f"Time: {time.perf_counter() - start:.2f}s")
    print(s)
