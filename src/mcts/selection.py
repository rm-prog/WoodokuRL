import jax
import jax.numpy as jnp
from functools import partial

from src.env.actions import clear_lines, step

@jax.jit
def ucb_score(tree, node_id, c=1.4):

    visits = tree["visits"]
    value_sum = tree["value_sum"]

    q = value_sum / (visits + 1e-8)

    total_visits = visits[node_id]

    u = c * jnp.sqrt(
        jnp.log(total_visits + 1)
        / (visits + 1e-8)
    )

    return q + u

@jax.jit
def select(tree, grid, root_tiles):

    def select_child(carry):
        node_id, grid, total_reward = carry

        def select_by_ucb(carry):
            node_id, grid, total_reward = carry

            children = tree["children_id"][node_id]
            scores = jnp.where(
                children >= 0,
                ucb_score(tree, node_id)[children],
                -jnp.inf,
            )

            child_id = children[jnp.argmax(scores)]

            # The action that led from node_id -> child_id
            action = tree["action_from_parent"][child_id]

            tile_idx = action // 81
            position = action % 81

            new_grid, _, reward = step(
                grid,
                root_tiles[tile_idx],
                position,
            )
            total_reward = total_reward + reward

            return child_id, new_grid, total_reward

        def unexpanded_node(carry):
            return carry

        should_descend = (
            jnp.any(tree["children_id"][node_id] >= 0)
            & (~tree["terminal"][node_id])
            & (tree["depth"][node_id] < 3)
        )

        return jax.lax.cond(
            should_descend,
            select_by_ucb,
            unexpanded_node,
            carry,
        )

    def cond(carry):
        node_id, grid, total_reward = carry

        return (
            jnp.any(tree["children_id"][node_id] >= 0)
            & (~tree["terminal"][node_id])
            & (tree["depth"][node_id] < 3)
        )

    current_node, selected_grid, total_reward = jax.lax.while_loop(
        cond,
        select_child,
        (0, grid),
    )

    return current_node, selected_grid, total_reward
