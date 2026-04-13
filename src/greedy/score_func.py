import jax
import jax.numpy as jnp
from src.config import BOX_SIZE

def zero_score(field):
    return 0

def empty_lines_score(field: jnp.array):
    row_empty = jnp.all(field == 0, axis=1)

    col_empty = jnp.all(field == 0, axis=0)

    boxes = jnp.array([
        field[i*BOX_SIZE:(i+1)*BOX_SIZE,
              j*BOX_SIZE:(j+1)*BOX_SIZE]
        for i in range(BOX_SIZE)
        for j in range(BOX_SIZE)
    ]).reshape(BOX_SIZE, BOX_SIZE, BOX_SIZE, BOX_SIZE)

    box_empty = jnp.all(boxes == 0, axis=(2, 3))

    return (
        jnp.sum(row_empty) +
        jnp.sum(col_empty) +
        jnp.sum(box_empty)
    )