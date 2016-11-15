# coding=utf-8
import nltk
import random
import csv
from parser import CsvReader
from nltk.corpus import stopwords

class Classifier:
    def __init__(self, target, train_set, test_set, classifier):
        # type: (basestring, list, list, classifier) -> object
        # Ex: ("Atheism", [],[], NaiveBayesClassifier)
        self.target = target
        self.train_set = train_set
        self.test_set = test_set
        self.classifier = classifier


if __name__ == '__main__':
    targets = ["Atheism", "Hillary Clinton", "Donald Trump", "Legalization of Abortion",
               "Climate Change is a Real Concern", "Feminist Movement"]
    stop_words = set(stopwords.words('english'))

    # Bag of Words
    def features(words):
        return dict([(word, True) for word in words])

    parser = CsvReader("data/StanceDataset/test.csv")
    tweets = parser.parse()
    tweets_by_target = [(filter(lambda x: x.target == t, tweets), t)for t in targets]

    for (tweets, target) in tweets_by_target:  # removendo stopwords
        for tweet in tweets:
            tweet.tweet = nltk.word_tokenize(tweet.tweet)

        featuresets = [(features(n), tag) for (n, tag) in [(tweet.tweet, tweet.stance) for tweet in tweets]]
        size = int(len(featuresets) * 0.25)
        train_set, test_set = featuresets[size:], featuresets[:size]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print 'accuracy: ', target, nltk.classify.accuracy(classifier, test_set)