import argparse
import time

import jax
import jax.numpy as jnp

from src.mcts.mcts import run_mcts


def make_test_state():
    grid = jnp.array([
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=jnp.int32)
    tiles = jnp.array([0, 5, 5], dtype=jnp.int32)
    return grid, tiles


def time_mcts(grid, tiles, batch_size):
    start = time.perf_counter()
    result = run_mcts(grid, tiles, batch_size=batch_size)
    jax.block_until_ready(result)
    return time.perf_counter() - start, result


def main():
    parser = argparse.ArgumentParser(description="Benchmark cold and warmed MCTS.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    grid, tiles = make_test_state()

    cold_seconds, result = time_mcts(grid, tiles, args.batch_size)
    print(f"device: {jax.devices()[0]}")
    print(f"batch size: {args.batch_size}")
    print(f"cold run, including JIT compilation: {cold_seconds:.3f}s")
    print(f"selected tile: {int(result[1])}, selected action: {int(result[2])}")

    warmup_seconds, _ = time_mcts(grid, tiles, args.batch_size)
    print(f"warmup run: {warmup_seconds:.3f}s")

    warm_seconds = []
    for _ in range(args.repeats):
        elapsed, _ = time_mcts(grid, tiles, args.batch_size)
        warm_seconds.append(elapsed)

    average = sum(warm_seconds) / len(warm_seconds)
    print("warmed runs: " + ", ".join(f"{value:.3f}s" for value in warm_seconds))
    print(f"average warmed run: {average:.3f}s")


if __name__ == "__main__":
    main()
