from generateVectors import generateTrainingVectors
from generateVectors import generateTestingVectors
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
    #pass
    distances = []
    for x in range(len(trainingVectors)):
		dist = euclideanDistance(testVector[1], trainingVectors[x][1])
		distances.append((trainingVectors[x], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []

    for x in range(k):
        neighbors.append(distances[x][0])

    # testing the sorted values
    #for dist in range(len(distances)):
    #    print(distances[dist][0][0] + " : " + str(distances[dist][1]))

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
    trainVectors = generateTrainingVectors()
    testVectors = generateTestingVectors()

    #print(euclideanDistance(testVectors[0][1], testVectors[1][1]))

    # computing the k-nearest neighbors for test document 6
    neighbors = computeKnn(k, testVectors[4], trainVectors)
   
    for neighbor in neighbors:
       print(neighbor[0])
    
    print("Predicted major is: " + getMajor(neighbors))