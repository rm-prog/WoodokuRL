import jax
import jax.numpy as jnp
from jax import lax


def backprop(tree, node, value):

    def cond_fn(state):
        node, _ = state
        return node != -1

    def body_fn(state):
        node, tree = state

        tree = tree.copy()

        tree["visits"] = tree["visits"].at[node].add(1)
        tree["value_sum"] = tree["value_sum"].at[node].add(value)

        parent = tree["parent"][node]

        return (parent, tree)

    _, tree = lax.while_loop(
        cond_fn,
        body_fn,
        (node, tree)
    )

    return tree