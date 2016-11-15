import csv


# Classe que modela o conjunto de treinamento/teste
class Data:
    def __init__(self, tweet, target, stance, opinion_towards, sentiment):
        # type: (basestring, basestring, string, string, basestring) -> object
        self.tweet = tweet
        self.target = target
        self.stance = stance
        self.opinionTowards = opinion_towards
        self.sentiment = sentiment


class CsvReader:
    def __init__(self, file_name):
        self.file_name = file_name

    def parse(self):
        file = open(self.file_name, 'rU')
        reader = csv.reader(file, delimiter=',')
        tweets = []
        for row in reader:
            tweets += [Data(row[0], row[1], row[2], row[3], row[4])]
        file.close()
        return tweets[1:]

#Teste
if __name__ == '__main__':
    c = CsvReader("data/StanceDataset/test.csv")
    print c.parse()[1].tweet