# Linear Regressor
Linear regressor ML model, based in numpy and pandas.
The challenge of this project is to implement the training algorithms from "scratch", using only the facilities of matrix operations powered by the numpy's library.

## Dependencies:
  - numpy >= 1.25.0
  - pandas >= 2.0.3

## Inference
The predictor of this linear regressor maps an n-dimensional entry **x** = (**x**_1, **x**_2, **x**_3, ..., **x**_n) in to a real value **ŷ**, by the function f:R^n -> R given by:

f(**x**) = w_0 + w_1 * **x**_1 + w_2 * **x**_2 + w_3 * **x**_3 + ... + w_n * **x**_n

Where the w_i factors are the weights of the regressor

## Training Algorithm
This regressor implements the gradient descent algorithm to get the optimal weights w* = (w_0, w_1, w_2, ..., w_n). We start by an initial guess w^(0). 
Then, we iterativily update the guess with the bellow recursive formula

w^(n+1) = w^(n) - eta * grad L (w^(n))

Where 
  - w^(n) is the nth term of the sequence
  - eta = learning rate
  - grad L = gradient of the loss function

## How to use this project?
First, import your dataset, with pandas

```
dataset = DatasetHandler.get_dataframe(<path>)
```

Split it in to train an test datasets

```
train_dataset, test_dataset = DatasetHandler.train_test_split(dataset, <train/test ratio>)
```

Then, instantiate the linear regressor, with the training and the test datasets:

```
linear_reg = LinearRegressor(train_dataset, test_dataset)
```

Finally, train your model with your prefered method:

```
linear_reg.gradient_descendent_fit(<learning rate>, <iterations number>)
```

Now, the model is trained and you can infere the value of **y** for any given instance **x**:

```
linear_reg.predict(x)
```

You can also get the loss over the test dataset:

```
linear_reg.get_loss()
```

Done!
