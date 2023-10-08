import numpy as np
import pandas as pd
from dataset_handler import DatasetHandler

class LinearRegressor:
    def __init__(self, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame) -> None:
        """Construtor padrão do regressor linear

        Args:
            train_dataset (pd.DataFrame): dataset de treino
            test_dataset (pd.DataFrame): dataset de teste
        """
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.x_dim = train_dataset.shape[1] - 1
        self.weights_dim = self.x_dim + 1
        self.weights = np.zeros(self.weights_dim)
        self.dimension_size = len(self.weights) - 1
    
    
    def gradient_descendent_fit(self, learning_rate: float, n_iterations: int) -> None:
        """Aplicação do algoritmo de descida em gradiente

        Args:
            learning_rate (float): taxa de aprendizado
            n_iterations (int): número de iterações
        """
        for i in range(n_iterations):
            # w^(t+1) = w^(t) - eta * grad(L(w^(t)))
            self.weights = self.weights - learning_rate*self.loss_gradient()
            print(self.weights)
            # self.weight_gradient()
    
    
    def loss_gradient(self) -> np.array:
        """Cálculo do gradiente da função de custo

        Returns:
            np.array: gradiente da função de custo, em w = (w_0, w_1, ..., w_n)
        """
        # grad = (grad_w_0, grad_w_1, grad_w_2, ..., grad_w_n)
        grad = np.zeros(self.weights_dim)
        
        # pontos do dataset de treino
        points = self.train_dataset.index
        n_points = len(points)
        
        for i in points:
            point = np.array(self.train_dataset.loc[i])
            w_0 = self.weights[0]
            y = point[:-1][0]
            dot_prod = np.dot(point[:-1], self.weights[1:])
            
            grad[0] = grad[0] + (2*(w_0 + dot_prod - y)/n_points)
            
            for i in range(1, self.weights_dim):
                grad[i] = grad[i] + (2*point[i]*(w_0 + dot_prod - y)/n_points)
         
        return grad
    
    
    def predict(self, x_coordinates: np.array) -> float:
        """Prediz o valor da variável dependente y com base no valor da variável
        dependete x = (x_1, x_2, x_3, ..., x_n) e nos pesos do regressor linear
        w = (w_0, w_1, w_2, ..., w_n)

        Args:
            x_coordinates (np.array): coordenadas de x = (x_1, x_2, x_3, ..., x_n)

        Returns:
            float: valor predito ŷ = ŷ(x)
        """
        y = self.weights[0] + np.dot(self.weights[1:], x_coordinates)
    
    
def test():
    df = DatasetHandler.get_dataframe('01.csv')
    train_dataset, test_dataset = DatasetHandler.train_test_split(df, 0.8)
    lin_reg = LinearRegressor(train_dataset, test_dataset)
    
    lin_reg.gradient_descendent_fit(0.015, 40)
    # print(lin_reg.weights)
    
test()