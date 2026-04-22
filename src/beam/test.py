import jax
from functools import partial
import time

from src.sim.simulation import simulate_games
from src.greedy.score_func import empty_lines_score
from src.beam.beam_search import beam_search_with_order

beam_empty_lines = partial(beam_search_with_order, score_fn=empty_lines_score)

key = jax.random.key(0)
key1 = jax.random.key(1)
key2 = jax.random.key(2)
key3 = jax.random.key(3)

beam_5_games = partial(simulate_games, num_games=5, func=beam_empty_lines)

for k in [key, key1, key2, key3]:
    start = time.perf_counter()
    s = beam_5_games(k).block_until_ready()
    print(f"Time: {time.perf_counter() - start:.2f}s")
    print(s)