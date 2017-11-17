from generateVectors import generateVectors
from collections import defaultdict
from agglomerative import agglomerative
from knn import *

if __name__ == '__main__':
    k = 5
    allVectors = generateVectors()

    agglomerative(allVectors)
    #  Fill out solutionVectors
    slnVectors = defaultdict(list)
    for vectorClass, vector in allVectors:
        slnVectors[vectorClass].append(vector)

    # Fill out classifiedVectors
    classifiedVectors = defaultdict(list)
    for i in range(len(allVectors)):
        vector = allVectors.pop(0)  # Remove vector from dataset

        # replace the last parameter of computeKnn with the desired distance formula
        neighbors = computeKnn(k, vector, allVectors, euclideanDistance)
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
    