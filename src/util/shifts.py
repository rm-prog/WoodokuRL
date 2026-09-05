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

@jax.jit
def shift_tile(tile, row, col):

    rows = jnp.arange(9)[:, None]
    cols = jnp.arange(9)[None, :]

    src_rows = rows - row
    src_cols = cols - col

    valid = (
        (src_rows >= 0) &
        (src_rows < 9) &
        (src_cols >= 0) &
        (src_cols < 9)
    )

    src_rows = jnp.clip(src_rows, 0, 8)
    src_cols = jnp.clip(src_cols, 0, 8)

    shifted = tile[src_rows, src_cols]

    return jnp.where(valid, shifted, 0)