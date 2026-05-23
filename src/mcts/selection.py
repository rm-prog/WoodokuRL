import jax
import jax.numpy as jnp

def ucb_score(tree, idx, c=1.4):

    q = tree["value_sum"][idx] / (tree["visits"][idx] + 1e-8)

    total_visits = jnp.sum(tree["visits"])

    u = c * jnp.sqrt(
        jnp.log(total_visits + 1)
        / (tree["visits"][idx] + 1e-8)
    )

    return q + u


def select(tree):

    idxs = jnp.arange(tree["visits"].shape[0])

    scores = jax.vmap(
        lambda i: ucb_score(tree, i)
    )(idxs)

    scores = jnp.where(
        tree["valid"],
        scores,
        -jnp.inf
    )

    return jnp.argmax(scores)