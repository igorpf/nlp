# coding=utf-8
import nltk
from nltk import word_tokenize
from parser import CsvReader
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import itertools

class Classifier:
    def __init__(self, target, train_set, test_set, classifier):
        # type: (basestring, list, list, classifier) -> object
        # Ex: ("Atheism", [],[], NaiveBayesClassifier)
        self.target = target
        self.train_set = train_set
        self.test_set = test_set
        self.classifier = classifier


if __name__ == '__main__':
    targets = ["Atheism", "Hillary Clinton", "Legalization of Abortion",
               "Climate Change is a Real Concern", "Feminist Movement"]
    option = 4
    # verificar se elemento pertence ao conjunto é mais rápido do que com listas (O(1) vs O(n))
    stop_words = set(stopwords.words('english'))

    sia = SentimentIntensityAnalyzer()
    def features(words, orig):
        if option == 1: # Bag of Words
            return dict([(word, True) for word in words if word not in stop_words])
        if option == 2: # Bigrams
            bigram_finder = BigramCollocationFinder.from_words(words)
            bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 10)
            return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
        if option == 3: # Análise de sentimentos
            f = {}
            ss = sia.polarity_scores(orig)
            # tirando o score de compound, pois ele nada revela sobre a polaridade em si
            score = sorted(filter(lambda x: x != 'compound', ss), key=ss.get, reverse=True)
            f["sentiment"] = score[0]
            return f


    parser = CsvReader("data/StanceDataset/test.csv")
    tweets = parser.parse()
    tweets_by_target = [(filter(lambda x: x.target == t, tweets), t)for t in targets]

    for (tweets, target) in tweets_by_target:
        for tweet in tweets:
            tweet.tweet = word_tokenize(tweet.tweet)
        featuresets = [(features(n, o), tag) for (n, tag, o) in [(tweet.tweet, tweet.stance, tweet.original_tweet) for tweet in tweets]]
        size = int(len(featuresets) * 0.05)
        train_set, test_set = featuresets[size:], featuresets[:size]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        classifier.show_most_informative_features()
        print 'accuracy: ', target, nltk.classify.accuracy(classifier, test_set)