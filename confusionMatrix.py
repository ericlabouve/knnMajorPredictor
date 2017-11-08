from collections import defaultdict

class ConfusionMatrix:
    def __init__(self):
        self.tp = 0  # True Positive
        self.fp = 0  # False Positive
        self.fn = 0  # False Negative - Knn never uses
        self.tn = 0  # True Negative - Knn never uses
        self.total = 0


    def add(self, actual, classified):
        # True positive
        if actual == classified:
            self.tp += 1
        # False positive
        elif actual != classified:
            self.fp += 1
        self.total += 1


    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / self.total

    def fscore(self):
        return (2 * self.precision() * self.recall()) / (self.precision() + self.recall())