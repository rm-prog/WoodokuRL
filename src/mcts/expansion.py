import jax
import jax.numpy as jnp

from src.env.actions import is_valid_placement_flat, step
from src.mcts.tree import add_child, update_node
from src.config import NUM_PLACEMENT_TRIPLES

@jax.jit
def expand(tree, idx, grid, tiles, key):

    def expand_node(new_tree, node_id, action):
        # Create a new child node
        child_id = new_tree["next_free"]
        tile_idx = action // 81
        available_tiles = new_tree["tiles_available"][node_id].at[tile_idx].set(False)
        new_tree = add_child(new_tree, node_id, action, new_tree["depth"][node_id] + 1, child_id, available_tiles)

        return new_tree

    def illegal_child(new_tree):
        return new_tree

    def valid_tile_placement(grid, action):
        tile_idx = action // 81
        placement_idx = action % 81
        return (is_valid_placement_flat(grid, tiles[tile_idx], placement_idx) 
                & (tree["tiles_available"][idx][tile_idx] == True))


    actions = jnp.arange(243)  # 3 tiles * 81 positions

    tree, _ = jax.lax.scan(
        lambda new_tree, action: (
            jax.lax.cond(
                valid_tile_placement(grid, action),
                lambda new_tree: expand_node(new_tree, idx, action),
                illegal_child,
                new_tree,
            ),
            None,
        ),
        tree,
        actions,
    )

    children = tree["children_id"][idx]
    has_children = jnp.any(children != -1)
    num_children = jnp.sum(children != -1)
    valid_children = children != -1

    tree["terminal"] = tree["terminal"].at[idx].set(~has_children)

    def choose_child(key):
        key, subkey = jax.random.split(key)

        random_position = jax.random.randint(
            subkey,
            (),
            0,
            num_children,
        )

        child_positions = jnp.where(
            valid_children,
            size=children.shape[0],
            fill_value=0,
        )[0]

        return children[child_positions[random_position]]

    chosen_child = jax.lax.cond(
        has_children,
        lambda _: choose_child(key),
        lambda _: jnp.array(-1, dtype=jnp.int32),
        operand=None,
    )

    def child_found(_):
        chosen_action = tree["action_from_parent"][chosen_child]
        
        chosen_grid, _, reward = step(
            grid,
            tiles[chosen_action // 81],
            chosen_action % 81,
        )
        
        return tree, chosen_child, chosen_grid, reward

    def no_child_found(_):
        return tree, idx, grid, jnp.int32(0)

    return jax.lax.cond(
        chosen_child != -1,
        child_found,
        no_child_found,
        operand=None,
    )
    