import jax.numpy as jnp

class Tile:
    def __init__(self, grid: jnp.array):
        self.grid = grid

    @property
    def height(self):
        return self.grid.shape[0]

    @property
    def width(self):
        return self.grid.shape[1]
    
    @property
    def size(self):
        return jnp.sum(self.grid == 1)