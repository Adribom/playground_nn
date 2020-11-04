import random
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
import os
import pickle 
from scipy import ndimage
from numpy import asarray
from matplotlib.colors import hsv_to_rgb

X_train_set_orig = pickle.load(open('X_train_set.pickle', 'rb'))
y_train_set = pickle.load(open('y_train_set.pickle', 'rb'))
X_test_set_orig = pickle.load(open('X_test_set.pickle', 'rb'))
y_test_set = pickle.load(open('y_test_set.pickle', 'rb'))

m_train = y_train_set.shape[1]
m_test = y_test_set.shape[1]
num_px = X_train_set_orig.shape[1]

"""
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(X_train_set_orig.shape))
print ("train_set_y shape: " + str(y_train_set.shape))
print ("test_set_x shape: " + str(X_test_set_orig.shape))
print ("test_set_y shape: " + str(y_test_set.shape))
"""

#IMPORTANT: A method to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b*c*d, a) is to use:
# X_flatten = X.reshape(X.shape[0], -1).T      # X.T is the transpose of X

train_set_x_flatten = X_train_set_orig.reshape(X_train_set_orig.shape[0], -1).T
test_set_x_flatten = X_test_set_orig.reshape(X_test_set_orig.shape[0], -1).T 

"""
print ("\n\ntrain_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(y_train_set.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(y_test_set.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
"""

#Standardize the dataset
train_set_x = train_set_x_flatten/255 
test_set_x = test_set_x_flatten/255 


