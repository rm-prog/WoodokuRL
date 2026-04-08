import jax.numpy as jnp

def pad_to_9x9(tile: jnp.array):
    out = jnp.zeros((9, 9), dtype=jnp.int32)
    h, w = tile.shape
    out = out.at[:h, :w].set(tile)
    return out

def crop_tile(tile):
    rows = jnp.any(tile == 1, axis=1)
    cols = jnp.any(tile == 1, axis=0)

    row_indices = jnp.where(rows)[0]
    col_indices = jnp.where(cols)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        return tile

    r_min, r_max = row_indices[0], row_indices[-1]
    c_min, c_max = col_indices[0], col_indices[-1]

    return tile[r_min:r_max+1, c_min:c_max+1]

def tile_to_str(tile):
    cropped = crop_tile(tile)

    return [
        ["■" if cell == 1 else "." for cell in row]
        for row in cropped
    ]

def pad_tile(tile_str, height, width):
    padded = []

    for row in tile_str:
        padded.append(row + [" "] * (width - len(row)))

    for _ in range(height - len(tile_str)):
        padded.append([" "] * width)

    return padded


def print_tiles(tiles):
    tile_strs = [tile_to_str(t) for t in tiles]

    heights = [len(t) for t in tile_strs]
    widths = [len(t[0]) for t in tile_strs]

    max_h = max(heights)
    max_w = max(widths)

    padded_tiles = [
        pad_tile(t, max_h, max_w) for t in tile_strs
    ]

    num_tiles = len(tiles)

    print()

    for r in range(max_h):
        line = ""
        for i in range(num_tiles):
            line += " ".join(padded_tiles[i][r]) + "   "
        print(line)

    print()

TILE_PERMS = jnp.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ])

TILES = jnp.stack([
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
])