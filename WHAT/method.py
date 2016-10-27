from __future__ import division
from random import shuffle

temp_file_name = "temp_file.csv"


def temp_file_generation(header, listoflist):
    import csv
    with open(temp_file_name, 'wb') as fwrite:
        writer = csv.writer(fwrite, delimiter=',')
        writer.writerow(header)
        for l in listoflist:
            writer.writerow(l)
    fwrite.close()
    import pdb
    pdb.set_trace()


def where_clusterer(filename):
    """
    This is function accepts a file with rows(=records) and clusters it. This is FASTNAP + PCA
    :param filename: Pass in the filename with rows as valid configurations
    :return: List of Cluster. Each cluster has a [[cluster_number], [list of members]]
    """
    from utilities.Tools.methods1 import wrapper_createTbl
    # The Data has to be access using this attribute table._rows.cells
    transformed_table = [[int(z) for z in x.cells[:-1]] + x.cells[-1:] for x in wrapper_createTbl(filename)._rows]
    cluster_numbers = set(map(lambda x: x[-1], transformed_table))

    # separating clusters
    # the element looks like [clusterno, rows]
    cluster_table = []
    for number in cluster_numbers:
        cluster_table.append([number] + [filter(lambda x: x[-1] == number, transformed_table)])
    return cluster_table


def generate_test_data(data_filename):
    raw_content = open(data_filename, "r").readlines()
    header = raw_content[0]
    content = [map(float, rc.split(",")) for rc in raw_content[1:]]
    indexes = range(len(content))
    shuffle(indexes)

    breakpoint = int(0.4 * len(content))

    train_indexes = indexes[:breakpoint]
    test_indexes = indexes[breakpoint:]

    train_independent = []
    train_dependent = []

    test_independent = []
    test_dependent = []

    for ti in train_indexes:
        train_independent.append(map(float, content[ti].split(',')[:-1]))
        train_dependent.append(float(content[ti].split(',')[-1]))

    assert (len(train_independent) == len(train_dependent)), "something wrong"

    for ti in test_indexes:
        test_independent.append(map(float, content[ti].split(',')[:-1]))
        test_dependent.append(float(content[ti].split(',')[-1]))

    assert (len(test_independent) == len(test_dependent)), "something wrong"

    extract = [content[ti] for ti in train_indexes]
    temp_file_generation(header, extract)
    training, no_of_clusters = where_clusterer(temp_file_name)

    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    generate_test_data("./Data/Apache_AllMeasurements.csv")