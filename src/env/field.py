import jax.numpy as jnp

class Field:

    def __init__(self):
        self.grid = jnp.zeros((9,9), dtype=jnp.int2)
        self.size = 9
        self.box_size = 3

    def __init__(self, size):
        self.grid = jnp.zeros((size, size), dtype=jnp.int2)
        self.size = size
        self.box_size = size / 3