# K-means clustering algorithm assignment
# Author: M. T. Hedges

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn

from math import sqrt

from pylab import plot, ylim
from random import choice
from scipy.spatial import distance

np.random.seed(17)

def main():
    x = 0

### Generate set number of clusters each of set number of points
def cluster_gen():
    num_clusters = 3
    num_points = 20

    raw_points = []

    point_index = 0
    for i in range(num_clusters):
        x1 = np.random.randint(5,100)
        y1 = np.random.randint(5,100)
        for j in range(20):
            x_cor = np.random.normal(x1,2.5)
            y_cor = np.random.normal(y1,2.5)

            #points[0][point_index] = x_cor
            #points[1][point_index] = y_cor

            point = [0, x_cor, y_cor]
            raw_points.append(point)
            point_index += 1

    points = np.array(raw_points)
    return points

def main():
    gen_points = cluster_gen()
    print(gen_points)
    npoints = len(gen_points)
    print(npoints)
    x = np.array([0.]*npoints)
    y = np.array([0.]*npoints)
    points = []

    for i in range(npoints):
        x[i] = gen_points[i][1]
        y[i] = gen_points[i][2]
        point = [x[i],y[i]]
        points.append(point)
    points = np.array(points)

    ### Define number of clusters
    k = 3

    distances = distance.cdist(points,points,'euclidean')
    #print(np.amax(distances))
    #print(np.where(distances == distances.max()))

    p1 = (points[np.unravel_index(np.ndarray.argmax(distances), 
        distances.shape)[0]])
    p2 = (points[np.unravel_index(np.ndarray.argmax(distances), 
        distances.shape)[1]])

    centers.append(p1)
    centers.append(p2)

    p3 = ([ abs(p1[0] - p2[0])/2. , np.amax(y)] if (abs(p1[0] - p2[0]) > 
            abs(p1[1] - p2[1])) else ([ abs(p1[0] - p2[0])/2.]))
    centers.append(p3)

    centers = np.array(centers)
    #print(centers)

    for i in range(len(points)):
        clust_dist = []
        for j in range(k):
            dist = np.linalg.norm(points[i]-centers[j])
            clust_dist.append(dist)
        print(clust_dist)
        clust_dist = np.array(clust_dist)
        cluster_id = (np.where(clust_dist == clust_dist.min()))[0][0]
        gen_points[i][0] = cluster_id
        print(cluster_id)
        #print(np.where(clust_dist == clust_dist.min()))

    print(gen_points)
    plt.scatter(x, y)
    plt.show()

if __name__ == "__main__":
    main()
