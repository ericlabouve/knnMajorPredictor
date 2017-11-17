from generateVectors import generateVectors
from collections import defaultdict
import operator
import math

def getIntersection(v1, v2):
    keys_v1 = set(v1.keys())
    keys_v2 = set(v2.keys())
    intersection = keys_v1 & keys_v2  # terms that v1 and v2 share

    return intersection

def cosineSimilarity(v1, v2):
    intersection = getIntersection(v1, v2)
    dot_product = 0
    v1_mag = 0
    v2_mag = 0
    
    for key in intersection:
        dot_product += v1[key] * v2[key]

    for key in v1.keys():
        v1_mag += v1[key] ** 2

    for key in v2.keys():
        v2_mag += v2[key] ** 2
    
    #print(dot_product / math.sqrt(v1_mag * v2_mag))
    return dot_product / math.sqrt(v1_mag * v2_mag)
    


def euclideanDistance(v1, v2):
    # https://stackoverflow.com/questions/18554012/intersecting-two-dictionaries-in-python
    intersection = getIntersection(v1, v2)
    keys_v1 = set(v1.keys()) - intersection  # terms that are only contained in v1
    keys_v2 = set(v2.keys()) - intersection  # terms that are only contained in v2
    sum = 0
    for key in intersection:
        sum += (v1[key] - v2[key]) ** 2
    for key in keys_v1:
        sum += v1[key] ** 2
    for key in keys_v2:
        sum += v2[key] ** 2
    return sum ** 1/2


def computeKnn(k, testVector, trainingVectors, distanceFormula):
    # use a priority queue to order trainingVectors based on smallest distance to cmputeKnn
    distances = []
    for x in range(len(trainingVectors)):
        dist = distanceFormula(testVector[1], trainingVectors[x][1])
        distances.append((trainingVectors[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getMajor(neighbors):
    count = dict()
    for neighbor in range(len(neighbors)):
        if neighbors[neighbor][0] in count:
            count[neighbors[neighbor][0]] = count[neighbors[neighbor][0]] + 1
        else:
            count[neighbors[neighbor][0]] = 1
    return max(count, key=count.get)