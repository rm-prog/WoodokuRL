import jax
import jax.numpy as jnp
from src.config import GRID_SIZE, BOX_SIZE, BOX_AMOUNT
from src.env.tiles import SHIFTED_TILES, SHIFTED_TILES_VALID, TILES, TILE_SIZES


@jax.jit
def place_tile(grid, tile_idx, row, col):
    shifted = SHIFTED_TILES[tile_idx, row, col]
    new_grid = grid + shifted
    overlap_valid = jnp.all(new_grid <= 1)
    boundary_valid = SHIFTED_TILES_VALID[tile_idx, row, col]
    valid = overlap_valid & boundary_valid

    illegal_grid = jnp.full((9, 9), -1, dtype=jnp.int32)
    result = jnp.where(valid, new_grid, illegal_grid)
    return result, valid

@jax.jit
def step(grid, tile_idx, action):
    row = action // GRID_SIZE
    col = action % GRID_SIZE

    placed_grid, valid = place_tile(grid, tile_idx, row, col)

    row_full = jnp.all(placed_grid != 0, axis=1)
    col_full = jnp.all(placed_grid != 0, axis=0)

    boxes = jnp.array([
        placed_grid[i*BOX_SIZE:(i+1)*BOX_SIZE,
                     j*BOX_SIZE:(j+1)*BOX_SIZE]
        for i in range(BOX_SIZE)
        for j in range(BOX_SIZE)
    ]).reshape(BOX_SIZE, BOX_SIZE, BOX_SIZE, BOX_SIZE)
    box_full = jnp.all(boxes != 0, axis=(2, 3))

    reward = (
        jnp.sum(row_full) +
        jnp.sum(col_full) +
        jnp.sum(box_full)
    ) * 9 + TILE_SIZES[tile_idx]

    cleared_grid = clear_lines(placed_grid)

    return cleared_grid, valid, reward


@jax.jit
def is_valid_placement(grid, tile_idx, row, col):
    _, valid = place_tile(grid, tile_idx, row, col)
    return valid


@jax.jit
def is_valid_placement_flat(grid, tile_idx, action):
    row = action // GRID_SIZE
    col = action % GRID_SIZE
    return is_valid_placement(grid, tile_idx, row, col)


v_is_valid = jax.jit(jax.vmap(
    is_valid_placement_flat,
    in_axes=(None, None, 0)
))


@jax.jit
def apply_move(grid, tile_idx, row, col):
    result, _ = place_tile(grid, tile_idx, row, col)
    return result


@jax.jit
def apply_move_valid(grid, tile_idx, row, col):
    result, valid = place_tile(grid, tile_idx, row, col)
    return result, valid


@jax.jit
def apply_move_flat(grid, tile_idx, action):
    row = action // GRID_SIZE
    col = action % GRID_SIZE
    return apply_move(grid, tile_idx, row, col)


@jax.jit
def apply_move_flat_valid(grid, tile_idx, action):
    row = action // GRID_SIZE
    col = action % GRID_SIZE
    return apply_move_valid(grid, tile_idx, row, col)


@jax.jit
def has_valid_placements(grid, tile_idx):
    """All 81 placements checked in one shot via lookup + einsum,
    instead of vmap-ing place_tile (which would also run clear_lines
    81 times for no reason)."""
    shifted_tiles = SHIFTED_TILES[tile_idx]                   # (9, 9, 9, 9)
    overlap = jnp.einsum('rcij,ij->rc', shifted_tiles, grid)  # (9, 9)
    valids = ((overlap == 0) & SHIFTED_TILES_VALID[tile_idx]).reshape(-1)

    actions = jnp.arange(GRID_SIZE * GRID_SIZE)
    has_any = jnp.any(valids)
    valid_actions = jnp.where(valids, actions, -1)
    return has_any, valid_actions


@jax.jit
def clear_lines(field: jnp.array):
    row_full = jnp.all(field != 0, axis=1).astype(bool)
    col_full = jnp.all(field != 0, axis=0).astype(bool)

    boxes = jnp.array([
        field[i*BOX_SIZE:(i+1)*BOX_SIZE,
              j*BOX_SIZE:(j+1)*BOX_SIZE]
        for i in range(BOX_SIZE)
        for j in range(BOX_SIZE)
    ]).reshape(BOX_SIZE, BOX_SIZE, BOX_SIZE, BOX_SIZE)

    box_full = jnp.all(boxes != 0, axis=(2, 3))
    box_mask = jnp.kron(box_full, jnp.ones((BOX_SIZE, BOX_SIZE)))

    clear = (
        row_full[:, None] |
        col_full[None, :] |
        box_mask.astype(bool)
    )

    return jnp.where(clear, 0, field)