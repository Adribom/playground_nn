# Building the parts of our algorithm

## The main steps for building a Neural Network are:

- Define the model structure (such as number of input features)
- Initialize the model's parameters
- Loop:
  1. Calculate current loss (forward propagation)
  2. Calculate current gradient (backward propagation)
  3. Update parameters (gradient descent)

You often build 1-3 separately and integrate them into one function we call model().

## Passos para a Math da NN:

1. Computar a helper function(sigmoid, ReLU etc.) 
2. Inicializar parâmetros
3. Forward and Backward propagation (achar a cost function e os grdientes)
![Captura de tela 2020-11-22 133443](https://user-images.githubusercontent.com/50799373/99909486-849e1a00-2cc7-11eb-9e2e-7b57cf6a28b0.png)
4. Otimizar os parâmetros usando os gradientes(learn $w$ and $b$ by minimizing the cost function $J$. For a parameter $\theta$, the update rule is $ \theta = \theta - \alpha \text{ } d\theta$, where $\alpha$ is the learning rate.)


Tips:
- what does an artificial neuron do? Simply put, it calculates a “weighted sum” of its input, adds a bias and then decides whether it should be “fired” or not