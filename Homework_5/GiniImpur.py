import numpy as np


def gini(labels_dis):
    total = 0
    for l in labels_dis:
        total += l

    gini = 0
    for l in labels_dis:
        current_prob = l / total
        gini += current_prob * (1 - current_prob)

    return gini


labels = np.array([10, 1, 4])
print(gini(labels))
