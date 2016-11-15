import nltk
import random
import parser

def features(word):
    return {'last_letter': word[-1]}

featuresets = [(features(n), gender) for (n, gender) in []]
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)


