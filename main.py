import numpy as np
import math

def file_parser(file_name):
    table = np.empty([0,0])

    with open(file_name, 'r') as f:
        for line in f:
            terms = np.char.split(line.strip(), ',').tolist()
            if np.size(table) == 0:
                table = terms
            else:
                table = np.vstack([table, terms])

    return table

def calc_entropy(table):
    # Assumes last column is the output value. All other columns are attribute values.
    y = table[:, -1]
    values = np.unique(y)
    occurences = np.zeros(np.shape(values))

    H = 0

    for i in range(np.size(occurences)):
        occurences[i] = np.count_nonzero(y == values[i])
        H = H - occurences[i] / np.size(y) * math.log2(occurences[i] / np.size(y))

    return H

def calc_gini_index(table):
    # Assumes last column is the output value. All other columns are attribute values.
    y = table[:, -1]
    values = np.unique(y)
    occurences = np.zeros(np.shape(values))

    GI = 1

    for i in range(np.size(occurences)):
        occurences[i] = np.count_nonzero(y == values[i])
        GI = GI - (occurences[i] / np.size(y))**2

    return GI

def calc_maj_error(table):
    # Assumes last column is the output value. All other columns are attribute values.
    y = table[:, -1]
    values = np.unique(y)
    occurences = np.zeros(np.shape(values))

    ME = 1

    for i in range(np.size(occurences)):
        occurences[i] = np.count_nonzero(y == values[i])
        ME = min(ME, 1 - (occurences[i] / np.size(y)))

    return ME

def calc_information_gain(type, table, column):
    # type - 0 for entropy, 1 for gini index
    # column - attribute to calculate information gain of

    x = table[:, column]
    values = np.unique(x)

    if (type == 0):
        IG = calc_entropy(table)

    elif (type == 1):
        IG = calc_gini_index(table)

    elif (type == 2):
        IG = calc_maj_error(table)

    for val in values:
        subtable = table[table[:,column] == val]
        if (type == 0):
            IG = IG - np.size(subtable,0) / np.size(table,0) * calc_entropy(subtable)
        elif (type == 1):
            IG = IG - np.size(subtable,0) / np.size(table,0) * calc_gini_index(subtable)
        elif (type == 2):
            IG = IG - np.size(subtable, 0) / np.size(table, 0) * calc_maj_error(subtable)


    return IG

def find_best_tree_split(type, table):
    # Find variable with highest information gain

    i_max = 0
    ig_max = -1000

    for i in range(np.size(table,1)-1):
        ig = calc_information_gain(type, table, i)
        if ig > ig_max:
            i_max = i
            ig_max = ig

    return i_max

def most_common_value(n):
    occurences = np.zeros(np.size(n, 0))
    values = np.unique(n)

    i_max = 0
    occ_max = 0

    for i in range(np.size(values, 0)):
        occurences[i] = np.count_nonzero(n == values[i])

        if occurences[i] > occ_max:
            occ_max = occurences[i]
            i_max = i

    return values[i_max]


def id3(type, table, max_tree_depth, depth, attributes, attribute_vals, mcv):

    # Check if leaf node
    y = table[:, -1]

    if np.size(np.unique(y)) == 0:
        return {mcv} # Return most common value of whole table

    if np.size(np.unique(y)) == 1:
        return {y[0]} # Return only remaining value

    if depth == max_tree_depth:
        return {most_common_value(y)} # Return most common value remaining

    tree = {}

    # Create root node
    j = find_best_tree_split(type, table)
    values = attribute_vals[j]

    for val in values:
        subtable = table[table[:, j] == val]
        tree[attributes[j] + ':' + val] = id3(type, subtable, max_tree_depth, depth+1, attributes, attribute_vals, mcv)

    return tree

def predict_value(tree, attributes, test_values):
    test_cond = [attributes[i] + ':' + test_values[i] for i in range(np.size(test_values))]

    # print(test_cond)

    keys = list(tree.keys())
    subtree = tree

    while np.size(list(subtree)) > 1:
        keys = list(subtree.keys())
        for condition in test_cond:

            if condition in keys:
                subtree = subtree[condition]


    return list(subtree)[0]

def test_tree(tree, attributes, test_table):

    correct = 0
    incorrect = 0

    for row in test_table:
        x = row[0:-1]
        y = row[-1]

        result = predict_value(tree, attributes, x)

        if result == y:
            correct += 1
        else:
            incorrect += 1

    return correct/(incorrect+correct)

if __name__ == "__main__":

    # Parse files
    car4_table_train = file_parser("./car-4/train.csv")
    car4_table_test = file_parser("./car-4/test.csv")

    car4_attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    car4_attribute_values = [['vhigh', 'high', 'med', 'low'],
    ['vhigh', 'high', 'med', 'low'],
     ['2', '3', '4', '5more'],
    ['2', '4', 'more'],
    ['small', 'med', 'big'],
    ['low', 'med', 'high']]


    # Run all three with different tree heights:

    for i in range(1,7):
        car4_tree_entropy = id3(0, car4_table_train, i, 0, car4_attributes, car4_attribute_values,
                                most_common_value(car4_table_train[:,-1]))
        car4_tree_gini = id3(1, car4_table_train, i, 0, car4_attributes, car4_attribute_values,
                                most_common_value(car4_table_train[:, -1]))
        car4_tree_me = id3(2, car4_table_train, i, 0, car4_attributes, car4_attribute_values,
                             most_common_value(car4_table_train[:, -1]))

        print('---------- Tree Height' + str(i) + ' ----------')
        print('ENTR-Based ID3 w/ Tree Height: ' + str(i) + ' ' + str(test_tree(car4_tree_entropy, car4_attributes, np.vstack([car4_table_train, car4_table_test]))))
        print('GINI-Based ID3 w/ Tree Height: ' + str(i) + ' ' + str(test_tree(car4_tree_gini, car4_attributes, np.vstack([car4_table_train, car4_table_test]))))
        print('MAJR-Based ID3 w/ Tree Height: ' + str(i) + ' ' + str(test_tree(car4_tree_me, car4_attributes, np.vstack([car4_table_train, car4_table_test]))))
        print('\n')