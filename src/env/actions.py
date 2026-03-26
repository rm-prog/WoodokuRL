import jax
import jax.numpy as jnp
from util.shifts import shift_right, shift_down

@jax.jit
def place_tile(grid, tile: jnp.array, row: int, col: int):
    shifted = shift_right(shift_down(tile, row), col)
        
    new_grid = grid + shifted
    overlap_valid = jnp.all(new_grid <= 1)
        
    original_count = jnp.sum(tile)
    shifted_count = jnp.sum(shifted)
    boundary_valid = (original_count == shifted_count)
        
    valid = overlap_valid & boundary_valid
        
    result = jnp.where(valid, new_grid, grid)
        
    return result, valid
    
def is_valid_placement(grid, tile, row, col):
    _, valid = place_tile(grid, tile, row, col)
    return valid
    
def is_valid_placement_flat(grid, tile, action):
    row = action // 9
    col = action % 9
    return is_valid_placement(grid, tile, row, col)

v_is_valid = jax.vmap(
    is_valid_placement_flat,
    in_axes=(None, None, 0)
)

def apply_move(grid, tile, row, col):
    result, _ = place_tile(grid, tile, row, col)
    return result
    
def apply_move_flat(grid, tile, action):
    row = action // 9
    col = action % 9
    return apply_move(grid, tile, row, col)
    
def has_valid_placements(grid, tile):
    actions = jnp.arange(81)
    valids = v_is_valid(grid, tile, actions)
    has_any = jnp.any(valids)
    valid_actions = actions[valids]

    return has_any, valid_actions
            