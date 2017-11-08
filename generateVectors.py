import os
from collections import defaultdict

categories = ['aerospace', 'architectural', 'bioresourceAndAgricultural', 'computer', 'electrical', 'liberalArts', 'materials',  'mechanical']
noiseWordArray = ["a", "about", "above", "all", "along", "also", "although", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "e.g.", "either", "etc", "etc.", "even", "ever", "enough", "for", "from", "further", "get", "gets", "got", "had", "have", "hardly", "has", "hasn't", "having", "he", "hence", "her", "here", "hereby", "herein", "hereof", "hereon", "hereto", "herewith", "him", "his", "how", "however", "i", "i.e.", "if", "in", "into", "it", "it's", "its", "me", "more", "most", "mr", "my", "near", "nor", "now", "no", "not", "or", "on", "of", "onto", "other", "our", "out", "over", "really", "said", "same", "she", "should", "shouldn't", "since", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "thereby", "therefore", "therefrom", "therein", "thereof", "thereon", "thereto", "therewith", "these", "they", "this", "those", "through", "thus", "to", "too", "under", "until", "unto", "upon", "us", "very", "was", "wasn't", "we", "were", "what", "when", "where", "whereby", "wherein", "whether", "which", "while", "who", "whom", "whose", "why", "with", "without", "would", "you", "your", "yours", "yes"]


def getTfVector(path):
    termFreq = defaultdict(int)  # term frequency dictionary
    with open(path, 'r') as f:  # Open file
        for line in f:
            for word in line.split():
                if word not in noiseWordArray:
                    termFreq[word] += 1
    return termFreq


def generateTrainingVectors():
    trainVectors = []  # 2-tuple of category and term frequency dictionary vectors
    for dir, category in zip(['trainTextData/' + category + '/' for category in categories], categories):  # For all training directories
        for filename in os.listdir(dir):  # For each file in all directories
            trainVectors.append((category, getTfVector(dir + filename)))
    return trainVectors


def generateTestingVectors():
    testVectors = []  # 2-tuple of category and term frequency dictionary vectors
    for dir, category in zip(['testTextData/' + category + '/' for category in categories], categories):  # For all testing directories
        for filename in os.listdir(dir):  # For each file in all directories
            testVectors.append((category, getTfVector(dir + filename)))
    return testVectors

def generateVectors():
    return generateTrainingVectors() + generateTestingVectors()