import jax.numpy as jnp
import jax

def shift(tile, row, col):
    out = jnp.zeros((9, 9), dtype=tile.dtype)

    h, w = tile.shape

    def body(i, out):
        def inner(j, out):
            return out.at[row + i, col + j].set(tile[i, j])
        return jax.lax.fori_loop(0, w, inner, out)

    return jax.lax.fori_loop(0, h, body, out)