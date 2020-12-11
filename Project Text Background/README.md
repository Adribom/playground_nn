# Key steps:

- Initialize the parameters of the model
- Learn the parameters for the model by minimizing the cost  
- Use the learned parameters to make predictions (on the test set)
- Analyse the results and conclude

## General Architecture of the learning algorithm:

![General Architecture of the learning algorithm](https://user-images.githubusercontent.com/50799373/99905368-26fdd380-2caf-11eb-93d0-57b5450747af.png)
![Mathematical expression of the algorithm](https://user-images.githubusercontent.com/50799373/99905418-76440400-2caf-11eb-9614-69b2b26204ad.png)

## Building the parts of our algorithm

### The main steps for building a Neural Network are:

- Define the model structure (such as number of input features)
- Initialize the model's parameters
- Loop:
  1. Calculate current loss (forward propagation)
  2. Calculate current gradient (backward propagation)
  3. Update parameters (gradient descent)

You often build 1-3 separately and integrate them into one function we call model().