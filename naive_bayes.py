from sklearn.naive_bayes import MultinomialNB
class NaiveBayes():
    def __init__(self) -> None:
        self.clf = MultinomialNB()
    def train(self, x, y):
        self.clf = self.clf.fit(x, y)
    def predict(self, x):
        return self.clf.predict(x)