import jax.numpy as jnp
import jax

@jax.jit
def shift_down(arr: jnp.array, k: int):
    return jnp.concatenate(
        [jnp.zeros((k, arr.shape[1])), arr[:-k]],
        axis=0
    )

@jax.jit
def shift_up(arr: jnp.array, k: int):
    return jnp.concatenate(
        [arr[k:], jnp.zeros((k, arr.shape[1]))],
        axis=0
    )

@jax.jit
def shift_right(arr: jnp.array, k: int):
    return jnp.concatenate(
        [jnp.zeros((arr.shape[0], k)), arr[:, :-k]],
        axis=1
    )

@jax.jit
def shift_left(arr: jnp.array, k: int):
    return jnp.concatenate(
        [arr[:, k:], jnp.zeros((arr.shape[0], k))],
        axis=1
    )