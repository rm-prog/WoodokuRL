import jax
import jax.numpy as jnp
from functools import partial
from src.sim.simulation import simulate_games
from src.greedy.greedy_agent import greedy_with_order
from src.greedy.score_func import zero_score

grid = jnp.zeros((9,9), dtype=int)
greedy_zero = partial(greedy_with_order, score_fn = zero_score)

key = jax.random.key(0)

scores = simulate_games(1, greedy_zero, key)

print(scores)
print(jnp.mean(scores))
