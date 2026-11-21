import time

import jax

from src.mcts.mcts import three_move_mcts
from src.sim.simulation import simulate_games


def main():
    key = jax.random.key(0)
    start = time.perf_counter()
    scores = simulate_games(key, num_games=1, func=three_move_mcts)
    scores.block_until_ready()
    elapsed = time.perf_counter() - start

    print(f"Score: {scores[0]}")
    print(f"Elapsed time (including JIT compilation): {elapsed:.2f}s")


if __name__ == "__main__":
    main()
