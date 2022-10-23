from sklearn import svm
class SVM():
    def __init__(self) -> None:
        self.clf = svm.SVC()
    def train(self, x, y):
        self.clf = self.clf.fit(x, y)
    def predict(self, x):
        return self.clf.predict(x)