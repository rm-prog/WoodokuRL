import jax.numpy as jnp
from src.config import GRID_SIZE, BOX_SIZE

class Field:

    def __init__(self, size: int = GRID_SIZE, box_size: int = BOX_SIZE):
        self.grid = jnp.zeros((size, size), dtype=jnp.int2)
        self.size = size
        self.box_size = box_size
    