import jax
import jax.numpy as jnp

from src.mcts.tree import (
    init_tree,
    update_node,
    add_child,
)

from src.env.actions import step

def run_mcts(grid: jnp.array, num_iters=50):
    tree = init_tree()
    root = 0

    tree = add_child(tree, -1, -1, root)
    tree = update_node(tree, root, 0.0)

    for _ in range(num_iters):
        tree = mcts_iteration(tree, root, grid)

def mcts_iteration(tree, root, grid):
    node = select(tree, root)
    