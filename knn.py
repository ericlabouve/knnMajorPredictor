from generateVectors import generateTrainingVectors
from generateVectors import generateTestingVectors


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
    #pass
    distances = []
    for x in range(len(trainingVectors)):
		dist = euclideanDistance(testVector, trainingVectors[x])
		distances.append((trainingVectors[x], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []

    for x in range(k):
        neighbors.append(distances[x][0])

    return neighbors


if __name__ == '__main__':
    trainVectors = generateTrainingVectors()
    testVectors = generateTestingVectors()

    #print(euclideanDistance(testVectors[0][1], testVectors[1][1]))
