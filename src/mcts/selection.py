import jax
import jax.numpy as jnp
from jax import lax

def ucb_score(tree, parent, child, c=1.4):

    q = (
        tree["value_sum"][child]
        / (tree["visits"][child] + 1e-8)
    )

    u = c * jnp.sqrt(
        jnp.log1p(tree["visits"][parent])
        / (tree["visits"][child] + 1e-8)
    )

    return q + u

def select(tree, root):

    def cond_fn(node):

        children = tree["children"][node]
        valid = children != -1

        return jnp.any(valid)

    def body_fn(node):

        children = tree["children"][node]

        valid = children != -1

        valid_children = jnp.where(
            valid,
            children,
            0
        )

        scores = jax.vmap(
            lambda child: ucb_score(tree, node, child)
        )(valid_children)

        scores = jnp.where(
            valid,
            scores,
            -jnp.inf
        )

        best_idx = jnp.argmax(scores)

        next_node = children[best_idx]

        return next_node

    leaf = lax.while_loop(
        cond_fn,
        body_fn,
        root
    )

    return leaf