import jax.numpy as jnp

def pad_to_9x9(tile):
    out = jnp.zeros((9, 9), dtype=jnp.int32)
    h, w = tile.shape
    out = out.at[:h, :w].set(tile)
    return out


TILES = [
    pad_to_9x9(jnp.array([[1]])),

    pad_to_9x9(jnp.array([[1, 1]])),
    pad_to_9x9(jnp.array([[1],
                          [1]])),

    pad_to_9x9(jnp.array([[1, 0],
                          [0, 1]])),
    pad_to_9x9(jnp.array([[0, 1],
                          [1, 0]])),

    pad_to_9x9(jnp.array([[1, 1, 1]])),
    pad_to_9x9(jnp.array([[1],
                          [1],
                          [1]])),

    pad_to_9x9(jnp.array([[1, 0],
                          [1, 1]])),
    pad_to_9x9(jnp.array([[0, 1],
                          [1, 1]])),
    pad_to_9x9(jnp.array([[1, 1],
                          [1, 0]])),
    pad_to_9x9(jnp.array([[1, 1],
                          [0, 1]])),

    pad_to_9x9(jnp.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 0, 1]])),
    pad_to_9x9(jnp.array([[0, 0, 1],
                          [0, 1, 0],
                          [1, 0, 0]])),

    pad_to_9x9(jnp.array([[1, 0],
                          [1, 1],
                          [1, 0]])),
    pad_to_9x9(jnp.array([[0, 1],
                          [1, 1],
                          [0, 1]])),

    pad_to_9x9(jnp.array([[0, 1, 0],
                          [1, 1, 1]])),
    pad_to_9x9(jnp.array([[1, 1, 1],
                          [0, 1, 0]])),

    pad_to_9x9(jnp.array([[1, 1],
                          [1, 1]])),

    pad_to_9x9(jnp.array([[1, 1, 1, 1]])),
    pad_to_9x9(jnp.array([[1],
                          [1],
                          [1],
                          [1]])),

    pad_to_9x9(jnp.array([[1, 1],
                          [1, 0],
                          [1, 0]])),
    pad_to_9x9(jnp.array([[1, 1],
                          [0, 1],
                          [0, 1]])),

    pad_to_9x9(jnp.array([[1, 0, 0],
                          [1, 1, 1]])),
    pad_to_9x9(jnp.array([[1, 1, 1],
                          [1, 0, 0]])),

    pad_to_9x9(jnp.array([[1, 0],
                          [1, 0],
                          [1, 1]])),
    pad_to_9x9(jnp.array([[0, 1],
                          [0, 1],
                          [1, 1]])),

    pad_to_9x9(jnp.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0],
                          [0, 0, 0, 1]])),

    pad_to_9x9(jnp.array([[0, 0, 0, 1],
                          [0, 0, 1, 0],
                          [0, 1, 0, 0],
                          [1, 0, 0, 0]])),

    pad_to_9x9(jnp.array([[0, 0, 1],
                          [1, 1, 1]])),
    pad_to_9x9(jnp.array([[1, 1, 1],
                          [0, 0, 1]])),

    pad_to_9x9(jnp.array([[1, 1, 0],
                          [0, 1, 1]])),
    pad_to_9x9(jnp.array([[0, 1, 1],
                          [1, 1, 0]])),

    pad_to_9x9(jnp.array([[0, 1],
                          [1, 1],
                          [1, 0]])),

    pad_to_9x9(jnp.array([[1, 0],
                          [1, 1],
                          [0, 1]])),

    pad_to_9x9(jnp.array([[1],
                          [1],
                          [1],
                          [1],
                          [1]])),

    pad_to_9x9(jnp.array([[1, 1, 1, 1, 1]])),

    pad_to_9x9(jnp.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]])),

    pad_to_9x9(jnp.array([[1, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 1]])),

    pad_to_9x9(jnp.array([[0, 0, 0, 0, 1],
                          [0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 0],
                          [1, 0, 0, 0, 0]])),

    pad_to_9x9(jnp.array([[1, 0, 0],
                          [1, 1, 1],
                          [1, 0, 0]])),

    pad_to_9x9(jnp.array([[0, 0, 1],
                          [1, 1, 1],
                          [0, 0, 1]])),

    pad_to_9x9(jnp.array([[0, 1, 0],
                          [0, 1, 0],
                          [1, 1, 1]])),

    pad_to_9x9(jnp.array([[1, 1, 1],
                          [0, 1, 0],
                          [0, 1, 0]])),

    pad_to_9x9(jnp.array([[1, 0, 0],
                          [1, 0, 0],
                          [1, 1, 1]])),

    pad_to_9x9(jnp.array([[0, 0, 1],
                          [0, 0, 1],
                          [1, 1, 1]])),

    pad_to_9x9(jnp.array([[1, 1, 1],
                          [1, 0, 0],
                          [1, 0, 0]])),

    pad_to_9x9(jnp.array([[1, 1, 1],
                          [0, 0, 1],
                          [0, 0, 1]])),

    pad_to_9x9(jnp.array([[1, 0, 1],
                          [1, 1, 1]])),

    pad_to_9x9(jnp.array([[1, 1, 1],
                          [1, 0, 1]])),

    pad_to_9x9(jnp.array([[1, 1],
                          [1, 0],
                          [1, 1]])),

    pad_to_9x9(jnp.array([[1, 1],
                          [0, 1],
                          [1, 1]])),
]