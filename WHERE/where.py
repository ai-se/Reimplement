from __future__ import division
from random import choice


def median_index(list):
    projected_values = [l.projection for l in list]
    if len(projected_values) < 1:
        assert(1==0), "wrong"
    if len(projected_values) %2 == 1:
        median_value = projected_values[((len(projected_values)+1)/2)-1]
    else:
        median_value = sum(projected_values[int(len(projected_values)/2)-1:int(len(projected_values)/2)+1])/2.0
    for i, pv in enumerate(projected_values):
        if pv > median_value: return i


def euclidean_distance(list1, list2):
    sum_sq = 0.0
    assert(len(list1) == len(list2)), "The length of the lists should be same"
    # add up the squared differences
    for i in range(len(list1)):
        sum_sq += (list1[i] - list2[i]) ** 2
    # take the square root of the result
    return sum_sq**0.5


class Item:
    """
    Item is used to store individual points
    """
    def __init__(self, id, decisions, objectives=None):
        self.id = id
        self.decisions = decisions
        self.objectives = objectives if type(objectives) is list else [objectives]
        self.projection = None


class Node:
    """
    Node is used to store the Node of the where tree. Nodes are the branching points
    """
    def __init__(self, level=0):
        self.level = level
        self.left = None
        self.right = None

    def set_left(self, items):
        # items is either a Node Object or a list of Item
        self.left = items

    def set_right(self, items):
        self.right = items


class WHERE:
    def __init__(self, csv_file):
        self.filename = csv_file
        self.content = self.get_content()
        self.items = [Item(i, c) for i, c in enumerate(self.content)]
        self.stopping_criteria = len(self.content) ** 0.5
        self.tree = None
        self.leaves = None
        self.run()

    def get_content(self):
        content = []
        with open(self.filename, 'r') as f:
            for line in f:
                content.append(map(float, line.split(',')))
        return content

    def assign_projection(self, items):
        def distant_point(chosen_item, other_items):
            distances = [euclidean_distance(chosen_item.decisions, item.decisions) for item in other_items]
            index_max_item = distances.index(max(distances))
            return other_items[index_max_item]

        def find_poles(passed_items):
            random_item = choice(passed_items)
            first_pole = distant_point(random_item, passed_items)
            second_pole = distant_point(first_pole, passed_items)
            return first_pole, second_pole

        def get_projection(item, pole1, pole2):
            c = euclidean_distance(pole1.decisions, pole2.decisions)
            a = euclidean_distance(item.decisions, pole1.decisions)
            b = euclidean_distance(item.decisions, pole2.decisions)
            return (a**2-b**2+c**2)/(2*c)

        assert(len(set([item.projection for item in items])) == 1), "All the elements should be None"
        pole1, pole2 = find_poles(items)
        for item in items:
            assert(item.projection is None), "Something is wrong"
            item.projection = get_projection(item, pole1, pole2)

        return items

    def split(self, level, items):
        if len(items) < self.stopping_criteria: return items
        else:
            items = self.assign_projection(items)
            sorted_items = sorted(items, key=lambda x: x.projection)
            median_value_projection = median_index(sorted_items)

            # removing projection numbers
            for item in sorted_items: item.projection = None

            new_node = Node(level+1)
            new_node.set_left(self.split(level+1, sorted_items[:median_value_projection]))
            new_node.set_right(self.split(level+1, sorted_items[median_value_projection:]))

        return new_node

    def retrieve_leaves(self, node):
        if type(node.left) is list:
            # print "!", len(node.left) + len(node.right)
            return [node.left, node.right]
        else:
            # print node.level, type(node.left), type(node.right)
            temp = []
            temp.extend(self.retrieve_leaves(node.left))
            temp.extend(self.retrieve_leaves(node.right))
            return temp

    def run(self):
        self.tree = self.split(0, self.items)
        self.leaves = self.retrieve_leaves(self.tree)


if __name__ == "__main__":
    filename = "./data.csv"
    leaves = WHERE(filename).leaves
    assert(sum([len(l) for l in leaves]) == 144), "Something is wrong"
    print len(leaves)


