from collections import defaultdict

class ConfusionMatrix:
    def __init__(self):
        self.tp = 0  # True Positive
        self.fp = 0  # False Positive - Knn never uses
        self.fn = 0  # False Negative
        self.tn = 0  # True Negative - Knn never uses
        self.total = 0

    def add(self, actual, classified):
        # True positive
        if actual == classified:
            self.tp += 1
        # False negative
        elif actual != classified:
            self.fn += 1

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def fscore(self):
        return (2 * self.precision() * self.recall()) / (self.precision() + self.recall())