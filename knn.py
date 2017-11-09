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


if __name__ == '__main__':
    k = 5
    allVectors = generateVectors()

    #  Fill out solutionVectors
    slnVectors = defaultdict(list)
    for vectorClass, vector in allVectors:
        slnVectors[vectorClass].append(vector)

    # Fill out classifiedVectors
    classifiedVectors = defaultdict(list)
    for i in range(len(allVectors)):
        vector = allVectors.pop(0)  # Remove vector from dataset

        # replace the last parameter of computeKnn with the desired distance formula
        neighbors = computeKnn(k, vector, allVectors, cosineSimilarity)
        allVectors.append(vector)  # Add vector back into dataset
        predictedClass = getMajor(neighbors)
        classifiedVectors[predictedClass].append(vector[1])
        print("A: " + vector[0] + " P: " + predictedClass)

    # ----Cross validation ----
    avgPrecision = 0
    avgRecall = 0
    avgFscore = 0
    for vectorClass, vectorList in classifiedVectors.items():
        # True positive
        tp = 0
        slnVectorList = slnVectors[vectorClass]
        for vector in vectorList:
            for slnVector in slnVectorList:
                if vector == slnVector:
                    tp += 1

        # False positive
        fp = len(vectorList) - tp

        # False Negative
        fn = len(slnVectorList) - tp

        # Precision
        precision = tp / (tp + fp)

        # Recall
        recall = tp / (tp + fn)

        # F-Score
        fscore = 0
        if precision + recall > 0:
            fscore = (2 * precision * recall) / (precision + recall)

        avgPrecision += precision
        avgRecall += recall
        avgFscore += fscore

        print(vectorClass)
        print("Precision: " + str(precision) + " Recall: " + str(recall) + " F-Score: " + str(fscore))

    avgPrecision /= len(slnVectors)
    avgRecall /= len(slnVectors)
    avgFscore /= len(slnVectors)
    print("Averages:")
    print("Precision: " + str(avgPrecision) + " Recall: " + str(avgRecall) + " F-Score: " + str(avgFscore))