# Linear Regressor
Linear regressor ML model, based in numpy and pandas
The challenge of this project is to implement the training algorithms from "scratch", using only the facilities of matrix operations powered by the numpy's library.

## Dependencies:
  - numpy >= 1.25.0
  - pandas >= 2.0.3

## Inference
The predictor of this linear regressor maps an n-dimensional entry **x** = (**x**_1, **x**_2, **x**_3, ..., **x**_n) in to a real value **ŷ**, by the function f:R^n -> R given by:

f(**x**) = w_0 + w_1***x**_1 + w_2***x**_2 + + w_3***x**_3 + ... + w_n***x**_n

Where the w_i factors are the weights of the regressor

## Training Algorithm
This regressor implements the gradient descent algorithm to get the optimal weights w* = (w_0, w_1, w_2, ..., w_n). We start by an initial guess w^(0). 
Then, we iterativily update the guess with the bellow recursive formula

w^(n+1) = w^(n) - eta*grad L (w^(n))

Where 
  - w^(n) is the nth term of the sequence
  - eta = learning rate
  - grad L = gradient of the loss function
