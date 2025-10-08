class perceptron:
    def __init__(self, n_features, learning_rate, max_epochs):
        self.weight = [0.0] * (n_features + 1)
        self.lr = learning_rate
        self.max_epochs = max_epochs

    #Helper in prediction and fitting
    def features_with_bias(self, x):
        return list(x) + [1.0]

    #Helper in prediction and fitting
    def decision_score(self, x):
        xb = self.features_with_bias(x)
        return sum(wi * xi for wi, xi in zip(self.weight, xb))  

    #Classify
    def predict(self, x):
        return 1 if self.decision_score(x) >= 0 else -1    

    # Function to train classifier
    def fit(self, train_data):
        epoch = 0
        while True:
            mistakes = 0
            for x, y in train_data:
                if y * self.decision_score(x) <= 0:         # misclassified or on boundary
                    xb = self.features_with_bias(x)
                    self.weight = [wi + self.lr * y * xi for wi, xi in zip(self.weight, xb)]
                    mistakes += 1
            epoch += 1
            if mistakes == 0 or epoch >= self.max_epochs:
                break
        return self

    #Evaluate accuracy
    def evaluate(self, dataset):
        correct = sum(1 for x, y in dataset if self.predict(x) == y)
        return correct / len(dataset)





