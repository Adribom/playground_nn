# Graded function: sigmoid

import numpy as np

def sigmoid(z):
    """ 
    Sigmoid takes a real value as input and outputs another value between 0 and 1. It’s easy to work with and has all the nice properties of activation functions: it’s non-linear, continuously differentiable, monotonic, and has a fixed output range.

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    return (1 / (1 + np.exp(-z)))

def sigmoid_prime(z):
    return sigmoid(z) * (1-sigmoid(z))




def tanh(z):
    """
    Tanh squashes a real-valued number to the range [-1, 1]. It’s non-linear. But unlike Sigmoid, its output is zero-centered. Therefore, in practice the tanh non-linearity is always preferred to the sigmoid nonlinearity.
    """

    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def tanh_prime(z):
	return 1 - np.power(tanh(z), 2)




def relu(z):
    """
    A recent invention which stands for Rectified Linear Units. The formula is deceptively simple: max(0,z). Despite its name and appearance, it’s not linear and provides the same benefits as Sigmoid but with better performance.
    """
    return np.clip(z, 0, z)

def relu_prime(z):
    return 1 if z > 0 else 0




def leakyrelu(z,alpha):
    """
    LeakyRelu is a variant of ReLU. Instead of being 0 when z<0, a leaky ReLU allows a small, non-zero, constant gradient α (Normally, α=0.01). However, the consistency of the benefit across tasks is presently unclear.
    """
    
    return max(alpha * z, z)

def leakyrelu_prime(z, alpha):
    return 1 if z > 0 else alpha




def elu(z, alpha):
    """
    Exponential Linear Unit or its widely known name ELU is a function that tend to converge cost to zero faster and produce more accurate results. Different to other activation functions, ELU has a extra alpha constant which should be positive number.

    ELU is very similiar to RELU except negative inputs. They are both in identity function form for non-negative inputs. On the other hand, ELU becomes smooth slowly until its output equal to -α whereas RELU sharply smoothes.
    """

    return z if z >= 0 else alpha*(e^z -1)

def elu_prime(z,alpha):
    return 1 if z > 0 else alpha*np.exp(z)