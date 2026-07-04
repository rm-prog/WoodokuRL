import jax
import jax.numpy as jnp

from src.mcts.tree import (
    init_candidates,
    update,
)
from src.mcts.selection import select
from src.mcts.rollout import rollout

from src.config import MCTS_ITERS

def q_value(tree):
    return tree["value_sum"] / (tree["visits"] + 1e-8)

@jax.jit
def run_mcts(grid, root_tiles):

    init_key = jax.random.key(42) 
    tree = init_candidates(grid, root_tiles, init_key)

    keys = jax.random.split(jax.random.key(0), MCTS_ITERS)

    def body(tree, key):
        tree = mcts_iteration(tree, key)
        return tree, None

    tree, _ = jax.lax.scan(body, tree, keys)

    q = q_value(tree)

    best_idx = jnp.argmax(q)

    return (
        tree["grid"][best_idx],
        tree["perm"][best_idx],
        tree["a1"][best_idx],
        tree["a2"][best_idx],
        tree["a3"][best_idx]
    )

def mcts_iteration(tree, key):
    
    idx = select(tree)

    value = rollout(tree["grid"][idx], key)

    tree = update(tree, idx, value)

    return tree
    