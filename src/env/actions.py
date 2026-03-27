import jax
import jax.numpy as jnp
from src.util.shifts import shift
from src.config import GRID_SIZE, BOX_SIZE, BOX_AMOUNT

@jax.jit
def place_tile(grid: jnp.array, tile: jnp.array, row: int, col: int):
    # shifted = shift_right(shift_down(tile, row), col)
    shifted = shift(tile, row, col)
        
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
    row = action // GRID_SIZE
    col = action % GRID_SIZE
    return is_valid_placement(grid, tile, row, col)

v_is_valid = jax.vmap(
    is_valid_placement_flat,
    in_axes=(None, None, 0)
)

def apply_move(grid, tile, row, col):
    result, _ = place_tile(grid, tile, row, col)
    return result
    
def apply_move_flat(grid, tile, action):
    row = action // GRID_SIZE
    col = action % GRID_SIZE
    return apply_move(grid, tile, row, col)
    
def has_valid_placements(grid, tile):
    actions = jnp.arange(GRID_SIZE*GRID_SIZE)
    valids = v_is_valid(grid, tile, actions)
    has_any = jnp.any(valids)
    valid_actions = jnp.where(valids, actions, -1)

    return has_any, valid_actions

@jax.jit
def clear_lines(field: jnp.array):
    row_full = jnp.all(field != 0, axis=1).astype(bool)
    col_full = jnp.all(field != 0, axis=0).astype(bool)

    boxes = field.reshape(BOX_SIZE, BOX_SIZE, BOX_SIZE, BOX_SIZE)
    box_full = jnp.all(boxes != 0, axis=(2, 3))

    box_mask = jnp.kron(box_full, jnp.ones((BOX_SIZE, BOX_SIZE)))

    clear = (
        row_full[:, None] |
        col_full[None, :] |
        box_mask.astype(bool)
    )

    return jnp.where(clear, 0, field)