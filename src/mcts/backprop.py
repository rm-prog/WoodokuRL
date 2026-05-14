def backprop(tree, node, value):

    while node != -1:

        tree["visits"] = tree["visits"].at[node].add(1)
        tree["value_sum"] = tree["value_sum"].at[node].add(value)

        node = tree["parent"][node]

    return tree