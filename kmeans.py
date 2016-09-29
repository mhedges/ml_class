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

    #points = np.array([[0.]* num_points * num_clusters ,[0.] * num_points * 
    #num_clusters])
    raw_points = []

    point_index = 0
    for i in range(num_clusters):
        x1 = random.randint(5,100)
        y1 = random.randint(5,100)
        for j in range(20):
            x_cor = random.normal(x1,2.5)
            y_cor = random.normal(y1,2.5)

            #points[0][point_index] = x_cor
            #points[1][point_index] = y_cor

            point = [0, x_cor, y_cor]
            raw_points.append(point)
            point_index += 1

    points = np.array(raw_points)
    return points

def kmeans():
    points = cluster_gen()
    print(points)
    npoints = len(points)
    print(npoints)
    x = np.array([0.]*npoints)
    y = np.array([0.]*npoints)

    for i in range(npoints):
        x[i] = points[i][1]
        y[i] = points[i][2]

    plt.scatter(x, y)
    plt.show()

def main():
    kmeans()

if __name__ == "__main__":
    main()
