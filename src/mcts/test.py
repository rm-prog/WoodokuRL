import jax
import jax.numpy as jnp

from src.mcts.tree import init_tree
from src.mcts.expansion import expand
from src.greedy.score_func import empty_lines_score

def make_dummy_grid():
    return jnp.zeros((9, 9), dtype=jnp.int32)


def make_dummy_tiles():
    return jnp.ones((3, 2, 2), dtype=jnp.int32)

def test_expand_creates_child():
    key = jax.random.key(0)

    tree = init_tree()
    grid = make_dummy_grid()
    tiles = make_dummy_tiles()

    tree = tree.copy()
    tree["grid"] = tree["grid"].at[0].set(grid)

    tree, child_id, child_grid = expand(
        tree=tree,
        node=0,
        tiles=tiles,
        key=key,
        score_fn=empty_lines_score,
    )

    assert child_id != 0
    print("child id: ", child_id)
    assert tree["parent"][child_id] == 0
    print("parent id is root: ", tree["parent"][child_id])

    assert jnp.any(tree["children"][0] == child_id)
    print("children of root: ", tree["children"][0])

    print("grid of child: ", tree["grid"][child_id])

    assert jnp.any(tree["grid"][child_id] >= 0)

test_expand_creates_child()