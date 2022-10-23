class Classifier():
    def __init__(self, algo) -> None:
        self.algo = algo

    def train(self, feature, y):
        self.algo.train(feature, y)
        
    def getAccuracy(self, x, y):
        correct =0
        for i, yi in enumerate(y):
            y_predicted = self.algo.predict([x[i]])
            if(y_predicted[0] == yi):
                correct+=1
        return (correct+0.0) / len(y)        
    