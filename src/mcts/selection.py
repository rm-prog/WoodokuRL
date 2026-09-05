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