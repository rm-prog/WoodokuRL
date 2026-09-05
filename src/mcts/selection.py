import jax
import jax.numpy as jnp
from functools import partial

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
def select(tree):
    valid = tree["valid"]
    unvisited = valid & (tree["visits"] == 0)

    def no_valid_move():
        return jnp.int32(-1)

    def select_unvisited():
        return jnp.argmax(unvisited)

    def select_by_ucb():
        scores = jnp.where(valid, ucb_score(tree), -jnp.inf)
        return jnp.argmax(scores)

    return jax.lax.cond(
        jnp.any(valid),
        lambda: jax.lax.cond(
            jnp.any(unvisited),
            select_unvisited,
            select_by_ucb,
        ),
        no_valid_move,
    )


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