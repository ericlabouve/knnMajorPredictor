""" This module represents calculations for 
    agglomerative clustering"""
import sys
from knn import euclideanDistance
from itertools import chain

class Matrix(object):
    """ Instance Variables:
            dimension: Number of clusters
            distances: Matrix of distance calculations for each cluster
                        represented as a dictionary of dictionaries
                        key -> cluster1
                        value -> dictionary in the form of {cluster2: distance(cluster1, cluster2)}
            categories: dictionary that map majors to lists, where the content of the list are the indices for the documents in the dataset for the major
        
        Notes:
            Clusters are formed through single link
    """
    def __init__(self, dataset):
        self.dimension = len(dataset)

        self.distances = {}
        self.createClusters(dataset)

        self.categories = {}
        self.mapCategories(dataset)

    # maps majors to their index in the dataset
    def mapCategories(self, dataset):
        index = 0

        for item in dataset:
            major = item[0]

            if major in self.categories:
                self.categories[major].append(index)
            else:
                self.categories[major] = [index]
            index += 1

    # create initial clusters and their distances
    def createClusters(self, dataset):
        for i in range(self.dimension):
            i_distances = {}
            
            for j in range(self.dimension):
                if i != j:
                    i_distances[(j)] = euclideanDistance(dataset[i][1], dataset[j][1])
            self.distances[(i)] = i_distances
        
    # Perform the single link, merge two closest clusters
    def singleLink(self, cluster1, cluster2):
        #print("cluster1: " + str(cluster1) + " cluster2: " + str(cluster2))
        newCluster = (cluster1, cluster2)
        newCluster_dist = {}

        cluster2_vals = self.distances.pop(cluster2)
        newCluster_dist = self.distances.pop(cluster1)

        newCluster_dist.pop(cluster1, None)
        newCluster_dist.pop(cluster2, None)

        for key in newCluster_dist:
            newCluster_dist[key] = min(newCluster_dist[key], cluster2_vals[key])
           
        self.distances[newCluster] = newCluster_dist
        self.updateValues(cluster1, cluster2, newCluster)
        self.dimension -= 1

    # update the values for cluster distances given:
    # the two closest clusters and the newcluster formed by them
    def updateValues(self, cluster1, cluster2, newCluster):
        for key, value in self.distances.items():
            if key != newCluster:
                value[newCluster] = self.distances[newCluster][key]
            if key != cluster1:
                value.pop(cluster1, None)
            if key != cluster2:
                value.pop(cluster2, None)

    # finds the shortest distance in the distance in the distance matrix
    # returns the two clusters with this distance
    def findShortestDistance(self):
        shortestDist = sys.float_info.max
        cluster1 = ()
        cluster2 = ()

        for key, value in self.distances.items():
            for subKey in value:
                if shortestDist > value[subKey]:
                    shortestDist = value[subKey]
                    cluster1 = key
                    cluster2 = subKey

        return (cluster1, cluster2)

    # converts the document index to the major it is related to
    def convertMatrixIndexToMajor(self, tup):
        docList = []

        for i, doc in enumerate(tup):
            major = 'None'
            if isinstance(tup[i], tuple):
                docList.append(self.convertMatrixIndexToMajor(tup[i]))

            for key, value in self.categories.items():
                if doc in value:
                    docList.append(key)
        return docList

    # prints the clusters in the matrix
    def printMatrix(self):
        for key, value in self.distances.items():
            print("Cluster: " + str(key))
            #print("Distances: " + str(value))

    # prints the cluster results
    def printResults(self, majorList):
        print(majorList)

# performs hierarchical agglomerative clustering
def agglomerative(dataset):
    matrix = Matrix(dataset)
    clusters = []
    
    while matrix.dimension > 1:
        cluster1, cluster2 = matrix.findShortestDistance()

        matrix.singleLink(cluster1, cluster2)

    #matrix.printMatrix()
    
    keys = matrix.distances.keys()
    results = matrix.convertMatrixIndexToMajor(keys)
    matrix.printResults(results)