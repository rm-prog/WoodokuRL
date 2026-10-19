import jax
import jax.numpy as jnp

from src.config import GRID_SIZE, MCTS_NODES, MCTS_MAX_CHILDREN
from src.env.actions import is_valid_placement_flat
from src.env.tiles import SHIFTED_TILES, SHIFTED_TILES_VALID



@jax.jit
def init_tree():

    return {
        "parent_id": -jnp.ones(MCTS_NODES, dtype=jnp.int32),
        "tiles_available": jnp.zeros((MCTS_NODES, 3), dtype=jnp.bool_),
        "action_from_parent": jnp.zeros(MCTS_NODES, dtype=jnp.int32),
        "children_id": -jnp.ones((MCTS_NODES, MCTS_MAX_CHILDREN), dtype=jnp.int32),
        "visits": jnp.zeros(MCTS_NODES, dtype=jnp.int32),
        "value_sum": jnp.zeros(MCTS_NODES, dtype=jnp.float32),
        "depth": -jnp.ones(MCTS_NODES, dtype=jnp.int32), 
        "terminal": jnp.zeros(MCTS_NODES, dtype=jnp.bool_),
        "next_free": jnp.array(1, dtype=jnp.int32),
    }


@jax.jit
def q_value(tree, indices=None):
    values = tree["value_sum"]
    visits = tree["visits"]
    q = values / (visits + 1e-8)
    return q if indices is None else q[indices]


@jax.jit
def update(tree, indices, values, selected):
    increments = selected.astype(jnp.int32)
    tree["visits"] = tree["visits"].at[indices].add(increments)
    tree["value_sum"] = tree["value_sum"].at[indices].add(
        jnp.where(selected, values, 0.0)
    )
    return tree


@jax.jit
def update_node(tree, node_id, value):
    tree["visits"] = tree["visits"].at[node_id].add(1)
    tree["value_sum"] = tree["value_sum"].at[node_id].add(value)
    return tree

@jax.jit
def add_child(tree, parent_id, action, depth, child_id, tiles_available):
    tree["children_id"] = tree["children_id"].at[parent_id, action].set(child_id)
    tree["parent_id"] = tree["parent_id"].at[child_id].set(parent_id)
    tree["action_from_parent"] = tree["action_from_parent"].at[child_id].set(action)
    tree["depth"] = tree["depth"].at[child_id].set(depth)
    tree["tiles_available"] = tree["tiles_available"].at[child_id].set(tiles_available)
    tree["next_free"] = tree["next_free"] + 1
    return tree
    