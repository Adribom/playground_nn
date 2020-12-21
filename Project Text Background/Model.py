import numpy as np
import matplotlib.pyplot as plt
import cv2
from Neural_Math import *
from Preprocessing_Data import *

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=True):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """

    w, b = InitializeParameters.initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = Optmization.optmize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = Optmization.predict(w, b, X_test)
    Y_prediction_train = Optmization.predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,"Y_prediction_test": Y_prediction_test, "Y_prediction_train" : Y_prediction_train, "w" : w, "b" : b,"learning_rate" : learning_rate,"num_iterations": num_iterations}
    
    return d


d = model(CheckData.train_set_x, CheckData.y_train_set, CheckData.test_set_x, CheckData.y_test_set, num_iterations = 3000, learning_rate = 0.005, print_cost = False)

index = 13
plt.imshow(CheckData.test_set_x[:,index].reshape((CheckData.num_px, CheckData.num_px, 3)))
print ("y = " + str(CheckData.y_test_set[0, index]) + ", you predicted that it is a \"" + DataManipulation.CATEGORIES[int(d["Y_prediction_test"][0, int(index)])] +  "\" picture.")
plt.show()

'''
# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()
'''
'''
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(CheckData.train_set_x, CheckData.y_train_set, CheckData.test_set_x, CheckData.y_test_set, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
'''