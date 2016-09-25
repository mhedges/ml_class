# K-means clustering algorithm assignment
# Author: M. T. Hedges

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn

from math import sqrt

from pylab import plot, ylim
from random import choice
from numpy import array, dot, random

random.seed(17)

def main():
    x = 0


def cluster_gen():
    num_clusters = 3
    num_points = 20

    points = np.array([[0.]* num_points * num_clusters ,[0.] * num_points * 
    num_clusters])
    x = np.array([0.] * num_points * num_clusters)
    y = np.array([0.] * num_points * num_clusters)

    point_index = 0
    for i in range(num_clusters):
        x1 = random.randint(5,100)
        y1 = random.randint(5,100)
        for j in range(20):
            x_cor = random.normal(x1,2.5)
            y_cor = random.normal(y1,2.5)
            points[0][point_index] = x_cor
            points[1][point_index] = y_cor
            x[point_index] = x_cor
            y[point_index] = y_cor
            point_index += 1

    return points

def kmeans():
    points = cluster_gen()
    print(points[0])
    print(points[1])

    plt.scatter(points[0], points[1])
    plt.show()

def main():
    kmeans()

if __name__ == "__main__":
    main()
