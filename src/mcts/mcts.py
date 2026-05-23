import jax
import jax.numpy as jnp
from jax import lax

from src.mcts.tree import (
    init_tree,
    update_node,
    add_child,
    init_root
)
from src.mcts.selection import select
from src.mcts.expansion import expand
from src.mcts.rollout import rollout
from src.mcts.backprop import backprop
from src.greedy.score_func import empty_lines_score
from src.sim.simulation import generate_tiles

from src.env.actions import step

def run_mcts(grid: jnp.array, root_tiles, num_iters=50):
    tree = init_tree()
    tree = init_root(tree, grid)
    root = 0

    key = jax.random.key(0)
    keys = jax.random.split(key, num_iters)

    for i in range(num_iters):
        tree = mcts_iteration(tree, root, root_tiles, keys[i])

    return tree

@jax.jit
def mcts_iteration(tree, root, root_tiles, key):
    
    node = select(tree, root)

    tiles = lax.cond(
        node == root,
        lambda _: root_tiles,
        lambda _: generate_tiles(key, tree["grid"][node]),
        operand=None
    )
    tree, child, child_grid = expand(tree, node, tiles, key)

    value = rollout(child_grid, key)

    tree = backprop(tree, child, value)

    return tree
    