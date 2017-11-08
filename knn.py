from generateVectors import generateTrainingVectors
from generateVectors import generateTestingVectors
from collections import defaultdict
from confusionMatrix import ConfusionMatrix
import operator



def euclideanDistance(v1, v2):
    # https://stackoverflow.com/questions/18554012/intersecting-two-dictionaries-in-python
    keys_v1 = set(v1.keys())
    keys_v2 = set(v2.keys())
    intersection = keys_v1 & keys_v2  # terms that v1 and v2 share
    keys_v1 -= intersection  # terms that are only contained in v1
    keys_v2 -= intersection  # terms that are only contained in v2
    sum = 0
    for key in intersection:
        sum += (v1[key] - v2[key]) ** 2
    for key in keys_v1:
        sum += v1[key] ** 2
    for key in keys_v2:
        sum += v2[key] ** 2
    return sum ** 1/2


def computeKnn(k, testVector, trainingVectors):
    # use a priority queue to order trainingVectors based on smallest distance to cmputeKnn
    distances = []
    for x in range(len(trainingVectors)):
        dist = euclideanDistance(testVector[1], trainingVectors[x][1])
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
    k = 7
    trainVectors = generateTrainingVectors()
    testVectors = generateTestingVectors()
    allVectors = trainVectors + testVectors

    # Cross validation
    confusionMatrixes = defaultdict(ConfusionMatrix)
    for i in range(len(allVectors)):
        vector = allVectors.pop(0)  # Remove vector from dataset
        neighbors = computeKnn(k, vector, allVectors)
        allVectors.append(vector)  # Add vector back into dataset
        prediction = getMajor(neighbors)
        confusionMatrixes[vector[0]].add(vector[0], prediction)
        #print("Predicted: " + prediction + "\tActual: " + vector[0])

    avgPrecision = 0
    avgRecall = 0
    avgFscore = 0
    for key, value in confusionMatrixes.items():
        print(key + ":")
        avgPrecision += value.precision()
        avgRecall += value.recall()
        avgFscore += value.fscore()
        print("Precision: " + str(value.precision()) + " Recall: " + str(value.recall()) + " F-Score: " + str(value.fscore()) + "\n")
    avgPrecision /= len(confusionMatrixes)
    avgRecall /= len(confusionMatrixes)
    avgFscore /= len(confusionMatrixes)
    print("Averages:")
    print("Precision: " + str(avgPrecision) + " Recall: " + str(avgRecall) + " F-Score: " + str(avgFscore))
