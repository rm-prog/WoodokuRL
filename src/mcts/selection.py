import jax
import jax.numpy as jnp

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

    scores = ucb_score(tree)

    scores = jnp.where(
        tree["valid"],
        scores,
        -jnp.inf
    )

    def no_valid_move():
        return jnp.int32(-1)

    def argmax_scores():
        return jnp.argmax(scores)

    return jax.lax.cond(
        jnp.any(tree["valid"]),
        argmax_scores,
        no_valid_move
    )