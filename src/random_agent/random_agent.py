import jax
import jax.numpy as jnp

def decide_moves(field: jnp.array, tiles):
    key = jax.random.key(42)