import numpy as np

def generate_data(weights: np.array, sample_size: np.array, limits: np.array, noise_std_dev: float) -> np.array:
    """Gera uma coleção de pontos, com distribuição linear, com pontos da forma
    P = (x_1, x_2, x_3, ..., x_n, y)
    Sendo y dado por
    y = w_o + w_1*x_1 + w_2*x_2 + w_3*x_3 + ... + w_n*x_n + E

    Args:
        weights (np.array): pesos da distribuição linear w = (w_0, w_1, x_2, ..., w_n)
        sample_size (np.array): tamanho da amostra
        limits (np.array): cotas inferiores e superiores, para cada coordenada do vetor (limits[0][i] < x_i < limits[1][i])
        noise_std_dev (float): desvio padrão do ruído E, com distribuição normal

    Returns:
        np.array: array com as coordenadas do ponto (x_1, x_2, ..., x_n)
    """
    dimension = len(weights) - 1
    points = []
    
    # geração de cada um dos n = sample_size pontos
    for i in range(sample_size):
        x_coor = generate_point(dimension, limits)
        y_coor = get_y(weights, x_coor) + generate_noise(0, noise_std_dev)
        
        # geração do novo ponto e adição à lista
        new_point = np.append(x_coor, y_coor)
        points.append(new_point)
    
    return np.array(points)


def get_y(weights: np.array, x_coordinates: np.array) -> float:
    """Obtém o valor da variável dependente y a partir da variável independente X = (x_1, x_2, ..., x_n)
    e dos pesos w = (w_0, w_1, w_2, ..., w_n), sendo
    y = w_o + w_1*x_1 + w_2*x_2 + w_3*x_3 + ... + w_n*x_n

    Args:
        weights (np.array): _description_
        x_coordinates (np.array): _description_

    Returns:
        float: _description_
    """
    y = weights[0] + np.dot(weights[1:], x_coordinates)
    return y


def generate_point(dimension: np.array, limits: np.array) -> np.array:
    """Gera um ponto em n-dimensional P = (x_1, x_2, x_3, ..., x_n) aleatório, cada qual
    com distribuição uniforme dentro dos limites (limits[0][i] < x_i < limits[1][i])

    Args:
        dimension (np.array): número n de dimensões
        limits (np.array): cotas inferiores e superiores de cada x_i do ponto P

    Returns:
        np.array: ponto gerado, em n-dimensões
    """
    # geração de pontos em (0,1)^n
    new_point = np.random.rand(dimension)
    
    # reescala e deslocamento dos pontos para os limites inferiores e superiores
    lower_lim, upper_lim = limits[0], limits[1]
    new_point = scale_and_shift(new_point, lower_lim, upper_lim)
    
    return new_point


def scale_and_shift(vector: np.array, lower_limits: np.array, upper_limits: np.array) -> np.array:
    """Escala e desloca cada coordenada v_i do vetor v = (v_1, v_2, v_3, ..., v_n) de 0 < v_i < 1, 
    para lower_limits[i] < v_i < upper_limits[i]

    Args:
        vector (np.array): vetor original
        lower_limits (np.array): cotas inferiores
        upper_limits (np.array): cotas superiores

    Returns:
        np.array: novo array, com coordenadas nos limites estabelecidos
    """
    # transformação da forma x_i <--- a + (b-a)*x_i
    scaled_vector = lower_limits + np.multiply((upper_limits - lower_limits), vector)
    return scaled_vector


def generate_noise(mean: float, std_dev: float) -> float:
    """Gera ruído no dado, com distribuição normal

    Args:
        mean (float): média da distribuição normal
        std_dev (float): desvio padrão da distribuição normal

    Returns:
        float: _description_
    """
    return np.random.normal(mean, std_dev)


def run_example():
    """Demonstração da geração de dados para um exemplo em particular
    """
    # w = (2, -1, 3, 5)
    # y = 2 + (-1)*x_1 + 3*x_2 + 5*x_3 + E
    weights = np.array([2, -1, 3, 5])
    
    # cotas inferiores = (0, 3, 1), cotas superiores = (1, 4, 2)
    limits = np.array([[0, 3, -2], [1, 4, 2]])
    
    # 8 pontos
    sample_size = 8
    
    # desvio padrão do ruido
    noise_std_dev = 2.5
    
    data = generate_data(weights, sample_size, limits, noise_std_dev)
    print(data)
    
    
run_example()