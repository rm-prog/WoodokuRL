import jax.numpy as jnp

class Field:

    def __init__(self, size: int = 9, box_size: int = 9):
        self.grid = jnp.zeros((size, size), dtype=jnp.int2)
        self.size = size
        self.box_size = box_size
    