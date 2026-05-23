import jax
import jax.numpy as jnp

from src.mcts.tree import (
    init_candidates,
    update
)
from src.mcts.selection import select
from src.mcts.rollout import rollout

def run_mcts(grid: jnp.array, root_tiles, num_iters=1024):

    init_key = jax.random.key(42)
    tree = init_candidates(grid, root_tiles, init_key)

    key = jax.random.key(0)
    keys = jax.random.split(key, num_iters)

    for i in range(num_iters):
        tree = mcts_iteration(tree, keys[i])

    return tree

@jax.jit
def mcts_iteration(tree, key):
    
    idx = select(tree)

    value = rollout(tree["grid"][idx], key)

    tree = update(tree, idx, value)

    return tree
    