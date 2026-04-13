import jax
import jax.numpy as jnp
import time
from functools import partial

from src.sim.simulation import simulate_games
from src.greedy.greedy_agent import greedy_with_order
from src.greedy.score_func import zero_score, empty_lines_score

greedy_zero_fn = partial(greedy_with_order, score_fn=zero_score)
greedy_empty_lines_fn = partial(greedy_with_order, score_fn=empty_lines_score)

key = jax.random.key(0)
key1 = jax.random.key(1)
key2 = jax.random.key(2)
key3 = jax.random.key(3)
key4 = jax.random.key(4)

greedy_zero_20_games = partial(simulate_games, num_games=20, func=greedy_zero_fn)

scores1 = greedy_zero_20_games(key).block_until_ready()
start = time.perf_counter()
scores2 = greedy_zero_20_games(key1).block_until_ready()
end = time.perf_counter()

print("Time for 20 games:", end - start)
print(scores1)
print(scores2)
print(jnp.mean(scores1+scores2))