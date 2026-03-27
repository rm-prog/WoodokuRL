import jax
import jax.numpy as jnp
from src.env.tiles import TILES, print_tiles
from src.env.actions import place_tile, has_valid_placements, clear_lines
from src.config import GRID_SIZE, BOX_SIZE
import time

class Game:
    def __init__(self):
        seed = int(time.time() * 1000) % (2**31)
        self.key = jax.random.key(seed)

    def start(self): 
        field = jnp.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        avail_tiles = self.generate_tiles(field)
        print("Woodoku game initialized\n")

        while True:
            print(field)
            if (len(avail_tiles) == 0): avail_tiles = self.generate_tiles(field)
            if (not self.any_tile_playable(field, avail_tiles)): 
                print("Game over. Cant place tiles\n")
                break

            print_tiles(avail_tiles)
            print("\nChoose a tile (0, 1, 2) or type -1 to quit:\n")
            tile_idx = int(input())
            print("\nEnter row and column (offset) to place tile (e.g., 0 0):\n")
            row = int(input())
            col = int(input())
            
            tile = avail_tiles[tile_idx]
            result, valid = place_tile(field, tile, row, col)
            if (valid):
                field = result
                print("Tile placed.\n")
                avail_tiles.pop(tile_idx)
                field = clear_lines(field)
            else:
                print("Invalid placement. Try again")

        print("Finished!")


    def any_tile_playable(self, field, tiles):
        tiles_stack = jnp.stack(tiles)

        def check_tile(tile):
            has_placement, _ = has_valid_placements(field, tile)
            return has_placement

        valids = jax.vmap(check_tile)(tiles_stack)

        return jnp.any(valids)

    def generate_tiles(self, field: jnp.array):
        result = []

        self.key, subkey = jax.random.split(self.key)
        perm = jax.random.permutation(subkey, len(TILES))

        for idx in perm:
            tile = TILES[int(idx)]
            has_placement, _ = has_valid_placements(field, tile)
            if has_placement:
                result.append(tile)
                break

        if (len(result) == 0): return []

        for _ in range(2):
            self.key, subkey = jax.random.split(self.key)
            idx = int(jax.random.randint(subkey, (), 0, len(TILES)))
            result.append(TILES[idx])

        return result
