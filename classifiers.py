from math import sqrt, exp

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.training_data = None
    
    def fit(self, X):
        self.training_data = X
    
    def euclidean_distance(self, row1, row2):
        distance = 0.0
        for i in range(len(row1) - 1):
            distance += (row1[i] - row2[i]) ** 2
        return sqrt(distance)
    
    def get_neighbors(self, test_row):
        distances = []
        for train_row in self.training_data:
            dist = self.euclidean_distance(test_row, train_row)
            distances.append((dist, train_row))
        distances.sort(key=lambda x: x[0])
        return distances[:self.k]
    
    def predict(self, X):
        predictions = []
        for row in X:
            neighbors = self.get_neighbors(row)
            output_values = [neighbor[1][-1] for neighbor in neighbors]
            prediction = max(set(output_values), key=output_values.count)
            predictions.append(prediction)
        return predictions

class SVMClassifier:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
    
    def fit(self, X):
        n_samples = len(X)
        n_features = len(X[0]) - 1
        
        self.w = [0] * n_features
        self.b = 0
        
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                features = x_i[:-1]
                y_i = 1 if x_i[-1] == 1 else -1
                
                condition = y_i * (sum(w_j * x_j for w_j, x_j in zip(self.w, features)) + self.b)
                
                if condition <= 1:
                    for i in range(n_features):
                        self.w[i] += self.lr * (y_i * features[i] - 2 * self.lambda_param * self.w[i])
                    self.b += self.lr * y_i
                else:
                    for i in range(n_features):
                        self.w[i] += self.lr * (-2 * self.lambda_param * self.w[i])
    
    def predict(self, X):
        predictions = []
        for x_i in X:
            features = x_i[:-1]
            prediction = sum(w_j * x_j for w_j, x_j in zip(self.w, features)) + self.b
            predictions.append(1.0 if prediction >= 0 else 0.0)
        return predictions

class GradientDescentClassifier:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    def fit(self, X):
        n_samples = len(X)
        n_features = len(X[0]) - 1
        
        self.weights = [0] * n_features
        self.bias = 0
        
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                features = x_i[:-1]
                y_i = x_i[-1]
                
                linear_pred = sum(w_j * x_j for w_j, x_j in zip(self.weights, features)) + self.bias
                y_predicted = self.sigmoid(linear_pred)
                
                # Update weights
                for i in range(n_features):
                    self.weights[i] -= self.lr * (y_predicted - y_i) * features[i]
                self.bias -= self.lr * (y_predicted - y_i)
    
    def predict(self, X):
        predictions = []
        for x_i in X:
            features = x_i[:-1]
            linear_pred = sum(w_j * x_j for w_j, x_j in zip(self.weights, features)) + self.bias
            y_predicted = self.sigmoid(linear_pred)
            predictions.append(1.0 if y_predicted >= 0.5 else 0.0)
        return predictions