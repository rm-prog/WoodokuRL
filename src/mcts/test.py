import jax
import jax.numpy as jnp

from src.mcts.mcts import run_mcts, mcts_iteration
from src.mcts.tree import init_candidates
from src.mcts.policy import random_policy
from src.env.tiles import TILES
from src.sim.simulation import simulate_games
from src.config import NUM_PLACEMENT_TRIPLES, NUM_ACTIONS

from functools import partial
import time

def make_dummy_grid():
    return jnp.zeros((9, 9), dtype=jnp.int32)


def make_dummy_tiles():
    return jnp.ones((3, 2, 2), dtype=jnp.int32)

def test_identical_tiles(grid):

    print("\n==============================")
    print("TEST 6: IDENTICAL TILES")
    print("==============================")

    L = jnp.array([
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 0],
    ], dtype=jnp.int32)

    cube = jnp.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ], dtype=jnp.int32)

    single_cell = jnp.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=jnp.int32)

    tiles = jnp.stack([
        L,
        cube,
        single_cell,
    ])

    tree = init_candidates(
        grid,
        tiles,
    )

    valid = tree["valid"]

    # Look at the first placement triple.
    # The six entries corresponding to its permutations
    # should have only permutation 0 surviving.
    print("\nFirst six permutation entries:")

    placement = 359558

    print(valid[placement], "\n", valid[NUM_PLACEMENT_TRIPLES+placement],
          "\n", valid[NUM_PLACEMENT_TRIPLES*2+placement], "\n", valid[NUM_PLACEMENT_TRIPLES*3+placement],
          "\n", valid[NUM_PLACEMENT_TRIPLES*4+placement], "\n", valid[NUM_PLACEMENT_TRIPLES*5+placement])

    p1 = placement // (NUM_ACTIONS ** 2)
    
    remainder = placement % (NUM_ACTIONS ** 2)
    
    p2 = remainder // NUM_ACTIONS
    
    p3 = remainder % NUM_ACTIONS

    print(p1, " ", p2, " ", p3)

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

test_identical_tiles(grid)

# tree = init_candidates(grid, jnp.stack([TILES[0], TILES[5], TILES[5]]))
# tree = mcts_iteration(tree, jax.random.key(0), grid, jnp.stack([TILES[0], TILES[5], TILES[5]]))
# jax.block_until_ready(tree)
# start = time.perf_counter()
# tree = mcts_iteration(tree, jax.random.key(0), grid, jnp.stack([TILES[0], TILES[5], TILES[5]]))
# jax.block_until_ready(tree)
# end = time.perf_counter()
# print("Why is this so slow: ", end-start)

# result = random_policy(grid, jnp.stack([TILES[0], TILES[5], TILES[5]]), jax.random.key(0))
# jax.block_until_ready(result)
# start = time.perf_counter()
# result = random_policy(grid, jnp.stack([TILES[0], TILES[5], TILES[5]]), jax.random.key(0))
# jax.block_until_ready(result)
# end = time.perf_counter()
# print("time for random policy: ", end-start)
# print(result)

result = run_mcts(make_dummy_grid(), jnp.stack([TILES[0], TILES[5], TILES[5]]))
jax.block_until_ready(result)
start = time.perf_counter()
result = run_mcts(make_dummy_grid(), jnp.stack([TILES[0], TILES[5], TILES[5]]))
jax.block_until_ready(result)
print(result)
end = time.perf_counter()
print("time for one turn: ", end-start)

# mcts_one_game = partial(simulate_games, num_games=1, func=run_mcts)
# key = jax.random.key(0)
# print(mcts_one_game(key))