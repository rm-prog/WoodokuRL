import jax
import jax.numpy as jnp
from functools import partial

from src.env.actions import apply_move_flat

@jax.jit
def ucb_score(tree, c=1.4):

    visits = tree["visits"]
    value_sum = tree["value_sum"]

    q = value_sum / (visits + 1e-8)

    total_visits = jnp.sum(visits)

    u = c * jnp.sqrt(
        jnp.log(total_visits + 1)
        / (visits + 1e-8)
    )

    return q + u

@jax.jit
def select(tree, grid, root_tiles):

    def select_child(carry):
        node_id, grid = carry

        def select_by_ucb(carry):
            node_id, grid = carry

            children = tree["children_id"][node_id]
            scores = jnp.where(
                children >= 0,
                ucb_score(tree)[children],
                -jnp.inf,
            )

            child_id = children[jnp.argmax(scores)]

            # The action that led from node_id -> child_id
            action = tree["action_from_parent"][child_id]

            tile_idx = action // 81
            position = action % 81

            new_grid = apply_move_flat(
                grid,
                root_tiles[tile_idx],
                position,
            )

            return child_id, new_grid

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
        node_id, grid = carry

        return (
            jnp.any(tree["children_id"][node_id] >= 0)
            & (~tree["terminal"][node_id])
            & (tree["depth"][node_id] < 3)
        )

    current_node, selected_grid = jax.lax.while_loop(
        cond,
        select_child,
        (0, grid),
    )

    return current_node, selected_grid

# @jax.jit
# def select(tree):
#     valid = tree["valid"]
#     unvisited = valid & (tree["visits"] == 0)

#     def no_valid_move():
#         return jnp.int32(-1)

#     def select_unvisited():
#         return jnp.argmax(unvisited)

#     def select_by_ucb():
#         scores = jnp.where(valid, ucb_score(tree), -jnp.inf)
#         return jnp.argmax(scores)

#     return jax.lax.cond(
#         jnp.any(valid),
#         lambda: jax.lax.cond(
#             jnp.any(unvisited),
#             select_unvisited,
#             select_by_ucb,
#         ),
#         no_valid_move,
#     )


@partial(jax.jit, static_argnames=["batch_size"])
def select_batch(tree, key, batch_size):
    valid = tree["valid"]
    unvisited = valid & (tree["visits"] == 0)

    def unvisited_scores():
        noise = jax.random.uniform(key, valid.shape)
        return jnp.where(unvisited, noise, -jnp.inf)

    def ucb_scores():
        return jnp.where(valid, ucb_score(tree), -jnp.inf)

    scores = jax.lax.cond(
        jnp.any(unvisited),
        unvisited_scores,
        ucb_scores,
    )
    _, indices = jax.lax.top_k(scores, batch_size)
    selected = jnp.isfinite(scores[indices])
    return indices, selected