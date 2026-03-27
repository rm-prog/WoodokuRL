import jax.numpy as jnp
import jax

@jax.jit
def shift(tile, row, col):
    out = jnp.zeros((9, 9), dtype=tile.dtype)

    h, w = tile.shape

    def body(i, out):
        def inner(j, out):
            return out.at[row + i, col + j].set(tile[i, j])
        return jax.lax.fori_loop(0, w, inner, out)

    return jax.lax.fori_loop(0, h, body, out)

@jax.jit(static_argnums=(1,))
def shift_down(arr: jnp.array, k: int):
    return jnp.concatenate(
        [jnp.zeros((k, arr.shape[1])), arr[:-k]],
        axis=0
    )

@jax.jit(static_argnums=(1,))
def shift_up(arr: jnp.array, k: int):
    return jnp.concatenate(
        [arr[k:], jnp.zeros((k, arr.shape[1]))],
        axis=0
    )

@jax.jit(static_argnums=(1,))
def shift_right(arr: jnp.array, k: int):
    return jnp.concatenate(
        [jnp.zeros((arr.shape[0], k)), arr[:, :-k]],
        axis=1
    )

@jax.jit(static_argnums=(1,))
def shift_left(arr: jnp.array, k: int):
    return jnp.concatenate(
        [arr[:, k:], jnp.zeros((arr.shape[0], k))],
        axis=1
    )