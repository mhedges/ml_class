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

np.random.seed()
#np.random.seed(17)

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

def clustering(centers, points, k):
    clust_dists = []
    for i in range(len(points)):
        clust_dist = []
        for j in range(k):
            dist = np.linalg.norm(points[i]-centers[j])
            clust_dist.append(dist)

        clust_dist = np.array(clust_dist)
        clust_dists.append(clust_dist)
        #cluster_id = (np.where(clust_dist == clust_dist.min()))[0][0]
        #gen_points[i][0] = cluster_id
    return clust_dists

def main():
    gen_points = cluster_gen()
    npoints = len(gen_points)
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

    centers = []

    ### "Smartly" generated centers
    #p1 = (points[np.unravel_index(np.ndarray.argmax(distances), 
    #    distances.shape)[0]])
    #p2 = (points[np.unravel_index(np.ndarray.argmax(distances), 
    #    distances.shape)[1]])

    #centers.append(p1)
    #centers.append(p2)

    #p3 = ([ abs(p1[0] - p2[0])/2. , np.amax(y)] if (abs(p1[0] - p2[0]) > 
    #        abs(p1[1] - p2[1])) else ([ abs(p1[0] - p2[0])/2.]))
    #centers.append(p3)

    ### Randomly generated centers
    p1 = [np.random.uniform(min(x), max(x)), np.random.uniform(min(y), max(y))]
    centers.append(p1)
    p2 = [np.random.uniform(min(x), max(x)), np.random.uniform(min(y), max(y))]
    centers.append(p2)
    p3 = [np.random.uniform(min(x), max(x)), np.random.uniform(min(y), max(y))]
    centers.append(p3)

    centers = np.array(centers)
    for x, y in centers:
        plt.scatter(x, y, color='orange')

    clust_dists = clustering(centers, points, k)

    for i in range(len(clust_dists)):
        clust_dist = clust_dists[i]
        cluster_id = (np.where(clust_dist == clust_dist.min()))[0][0]
        gen_points[i][0] = cluster_id

    centroids = np.array([[0.,0.]] * k)
    old_centroids = centers
    
    while np.array_equiv(old_centroids, centroids) == False :
        nums = [0]*k
        old_centroids = centroids
        centroids = np.array([[0.,0.]] * k)

        for k_id, x, y in gen_points:
            nums[int(k_id)] += 1
            centroids[int(k_id)][0] += x
            centroids[int(k_id)][1] += y

        for i in range(k):
            centroids[i][0] /= nums[i]
            centroids[i][1] /= nums[i]
        print(centroids)
        print(old_centroids)
        clust_dists = clustering(centroids, points, k)
        for i in range(len(clust_dists)):
            clust_dist = clust_dists[i]
            cluster_id = (np.where(clust_dist == clust_dist.min()))[0][0]
            gen_points[i][0] = cluster_id

        #input('well?')

    print('Algorithm converged for k = %i clusters' % k)
    print(gen_points)
    print(centroids)

    #for x, y in points:
    #    plt.scatter(x, y)
    for x, y in centroids:
        plt.scatter(x, y, color='red')
    for k_id, x, y in gen_points:
        if k_id == 0:
            plt.scatter(x,y, color='green')
        if k_id == 1:
            plt.scatter(x,y, color='black')
        if k_id == 2:
            plt.scatter(x,y, color='blue')
    plt.show()

if __name__ == "__main__":
    main()
