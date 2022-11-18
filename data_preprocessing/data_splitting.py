from sklearn.model_selection import train_test_split

class data_splitting():
    def __init__(self, x, y) -> None:
        self.X =x
        self.y = y
        self.TRAINING_RATIO = 0.8
        self.VALIDATION_RATIO = 0.25
        self.TESTING_RATIO = 0.2
    def data_splitting(self):
        X_train, X_test, y_train, y_test \
            = train_test_split(self.X, self.y, test_size = self.TESTING_RATIO, shuffle = True, random_state=8)
        X_train, X_val, y_train, y_val \
            = train_test_split(X_train, y_train, test_size= self.VALIDATION_RATIO, random_state=8)
        return X_train, X_val, X_test, y_train, y_val, y_test
        