import numpy as np
import pandas as pd
from dataset_handler import DatasetHandler

class LinearRegressor:
    def __init__(self, train_dataset, test_dataset) -> None:
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.x_dim = train_dataset.shape[1] - 1
        self.weights_dim = self.x_dim + 1
        self.weights = np.zeros(self.weights_dim)
        self.dimension_size = len(self.weights) - 1
    
    
    def gradient_descendent_fit(self, learning_rate, n_iterations):
        for i in range(n_iterations):
            # w^(t+1) = w^(t) - eta * grad(w^(t))
            self.weights = self.weights - learning_rate*self.weight_gradient()
            print(self.weights)
            # self.weight_gradient()
            pass
    
    
    def weight_gradient(self):
        # grad = (grad_w_0, grad_w_1, grad_w_2, ..., grad_w_n)
        
        grad = np.zeros(self.weights_dim)
        
        points = self.train_dataset.index
        n_points = len(points)
        
        for i in points:
            point = np.array(self.train_dataset.loc[i])
            w_0 = self.weights[0]
            y = point[:-1][0]
            dot_prod = np.dot(point[:-1], self.weights[1:])
            
            # w^(t+1) = w^(t) - eta * grad(w^(t))
            # self.weights = self.weights - learning_rate*self.weight_gradient()
            
            grad[0] = grad[0] + 2*(w_0 + dot_prod - y)/n_points
            
            for i in range(1, self.weights_dim):
                grad[i] = grad[i] + 2*point[i]*(w_0 + dot_prod - y)/n_points
         
        return grad
    
    
    def predict(self, x_coordinates: np.array) -> float:
        y = self.weights[0] + np.dot(self.weights[1:], x_coordinates)
    
    
def test():
    df = DatasetHandler.get_dataframe('01.csv')
    train_dataset, test_dataset = DatasetHandler.train_test_split(df, 0.8)
    lin_reg = LinearRegressor(train_dataset, test_dataset)
    
    lin_reg.gradient_descendent_fit(0.015, 80)
    # print(lin_reg.weights)
    
test()