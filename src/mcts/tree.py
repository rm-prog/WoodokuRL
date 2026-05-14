import jax
import jax.numpy as jnp
from src.config import GRID_SIZE

MAX_NODES = 5000
MAX_CHILDREN = 128
NUM_ACTIONS = GRID_SIZE * GRID_SIZE

def init_tree():

    tree = {
        "grid": jnp.zeros((MAX_NODES, GRID_SIZE, GRID_SIZE), dtype=jnp.int32),
        "visits": jnp.zeros((MAX_NODES,), dtype=jnp.int32),
        "value_sum": jnp.zeros((MAX_NODES,), dtype=jnp.float32),
        "parent": -jnp.ones((MAX_NODES,), dtype=jnp.int32),
        "children": -jnp.ones((MAX_NODES, MAX_CHILDREN), dtype=jnp.int32),
        "action_from_parent": -jnp.ones((MAX_NODES,), dtype=jnp.int32),
        "next_free": jnp.array(1, dtype=jnp.int32),
    }

    return tree

def q_value(tree, node_id):
    """Mean value of node."""
    return tree["value_sum"][node_id] / (tree["visits"][node_id] + 1e-8)


def is_expanded(tree, node_id):
    """Check if node has any children."""
    return jnp.any(tree["children"][node_id] != -1)


def get_children(tree, node_id):
    return tree["children"][node_id]


def get_parent(tree, node_id):
    return tree["parent"][node_id]

def add_child(tree, parent, action, child_id):

    children = tree["children"].at[parent, action].set(child_id)
    parent_arr = tree["parent"].at[child_id].set(parent)
    action_arr = tree["action_from_parent"].at[child_id].set(action)

    tree = tree.copy()
    tree["children"] = children
    tree["parent"] = parent_arr
    tree["action_from_parent"] = action_arr

    return tree

def update_node(tree, node_id, value):
    """
    Adds one simulation result to a node.
    """

    visits = tree["visits"].at[node_id].add(1)
    value_sum = tree["value_sum"].at[node_id].add(value)

    tree = tree.copy()
    tree["visits"] = visits
    tree["value_sum"] = value_sum

    return tree

def init_root(tree, grid):
    tree["grid"] = tree["grid"].at[0].set(grid)
    return tree