from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
class DecisionTree():
    
    def __init__(self, min_size,  max_depth ) -> None:
       self.dtree = DecisionTreeClassifier(min_samples_split= min_size, max_depth= max_depth)
    
    def train(self, x, y):
        self.dtree = self.dtree.fit(x, y)

    def predict(self, x):
       
        return self.dtree.predict(x)
        
    def validation(self, x_val, y_val):
        for x in x_val:
           print(self.dtree.predict(x))
    