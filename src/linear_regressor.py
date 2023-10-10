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
        """Aplicação do algoritmo de descida em gradiente para regressão linear

        Args:
            learning_rate (float): taxa de aprendizado
            n_iterations (int): número de iterações
        """
        for i in range(n_iterations):
            loss_grad = self.loss_gradient()
            
            # w^(t+1) = w^(t) - eta * grad(L(w^(t)))
            self.weights = self.weights - learning_rate*loss_grad
    
    
    def loss_gradient(self) -> np.array:
        """Cálculo do gradiente da função de custo

        Returns:
            np.array: gradiente grad L = (grad_w_0, grad_w_1, grad_w_2, ..., grad_w_n) da 
            função de custo L = sum((ŷ - y)^2)/n, em w = (w_0, w_1, ..., w_n)
        """
        # grad L = (grad_w_0, grad_w_1, grad_w_2, ..., grad_w_n)
        grad = np.zeros(self.weights_dim)
        
        # pontos do dataset de treino
        points = self.train_dataset.index
        n_points = len(points)
        
        # para cada ponto, computar sua parcela no gradiente e somar ao todo
        for i in points:
            # P = (x_1, x_2, x_3, ..., x_n, y)
            point = np.array(self.train_dataset.loc[i])
            
            # x = (x_1, x_2, x_3, ..., x_n); y = y
            x, y = point[:-1], point[-1]
            
            # ŷ = f(x) = w_0 + w_1*x_1 + w_2*x_2 + w_3*x_3 + ... + w_n*x_n
            y_hat = self.predict(x)
            
            # grad_0 = (2/n)*sum {ŷ_i - y_i}
            grad[0] = grad[0] + (2*(y_hat - y)/n_points)
            
            # grad_j = (2/n)*sum {x_j*(ŷ_i - y_i)}
            k = 0
            for grad_j in grad[1:]:
                grad_j = grad_j + (2*x[k]*(y_hat - y)/n_points)
                grad[k+1] = grad_j
                k = k + 1
            
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
        return y
    
    
    def get_loss(self):
        test_samples_indexes = self.test_dataset.index
        n_samples = len(test_samples_indexes)
        
        loss = 0
        
        for i in test_samples_indexes:
            sample = self.test_dataset.loc[i]
            x = sample[:-1]
            y = sample[-1]
            y_hat = self.predict(x)
             
            loss = loss + ((y_hat - y)*(y_hat - y)/n_samples)
        
        return loss
    
def test():
    df = DatasetHandler.get_dataframe('01.csv')
    train_dataset, test_dataset = DatasetHandler.train_test_split(df, 0.8)
    lin_reg = LinearRegressor(train_dataset, test_dataset)
    
    lin_reg.gradient_descendent_fit(0.01, 100)
    print(f'w = {lin_reg.weights}; loss = {lin_reg.get_loss()}')
    
test()