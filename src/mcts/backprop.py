import jax
import jax.numpy as jnp

from woodoku.src.mcts.tree import update_node

@jax.jit
def backprop(tree, node_id, value):

    def cond_fn(carry):
        tree, node_id = carry
        return node_id >= 0

    def body_fn(carry):
        tree, node_id = carry

        tree = update_node(tree, node_id, value)
        parent_id = tree["parent_id"][node_id]

        return tree, parent_id

    tree, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (tree, node_id),
    )

    return tree