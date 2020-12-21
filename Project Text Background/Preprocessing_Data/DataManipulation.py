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

#Compact variables of training data
TRAINDATADIR = "G:\\Meu Drive\\NN\\playground_nn\\Project Text Background\\Preprocessing_Data\\data\\train"
TESTDATADIR = "G:\\Meu Drive\\NN\\playground_nn\\Project Text Background\\Preprocessing_Data\\data\\test"
CATEGORIES = ["black", "white"]
training_data = []
test_data = []
IMG_SIZE = 50

#Variables for direct use of NN
X_train_set = []
y_train_set = []
X_test_set = []
y_test_set = []


def create_training_data():
    #Iterate throgh images directories (black amd white)
    for categorie in CATEGORIES:
        path = os.path.join(TRAINDATADIR, categorie)
        class_num = CATEGORIES.index(categorie) #get index of list to not use "white" and "black"
        for img in os.listdir(path):
            #try because there might be data error
            try:
                inverted_img_array = cv2.imread(os.path.join(path, img))  
                img_array = cv2.cvtColor(inverted_img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            #If error is found, pass
            except Exception as e:
                pass

def create_test_data():
    #Iterate throgh images directories (black and white)
    for categorie in CATEGORIES:
        path = os.path.join(TESTDATADIR, categorie)
        class_num = CATEGORIES.index(categorie) #get index of list to not use "black" and "white"
        for img in os.listdir(path):
            #try because there might be data error
            try:
                inverted_img_array = cv2.imread(os.path.join(path, img))  
                img_array = cv2.cvtColor(inverted_img_array, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                test_data.append([new_array, class_num])
            #If error is found, pass
            except Exception as e:
                pass


def save_training_data(X, y):
# Use pickle to store the process and not need to rebuild entire data set 
    pickle_out = open("X_train_set.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()  

    pickle_out = open("y_train_set.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

def save_test_data(X, y):
# Use pickle to store the process and not need to rebuild entire data set 
    pickle_out = open("X_test_set.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()  

    pickle_out = open("y_test_set.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()



                                                                            #WITH NUMPY:
                                                                            #np.save('features.npy',X) #saving
                                                                            #X=np.load('features.npy') #loading


# If I want to read back in:
    #pickle_in = open("X.pickle", "rb")
    #X = pickle.load(pickle_in)




create_training_data()
create_test_data()
#print(len(training_data))
random.shuffle(test_data)
random.shuffle(training_data)
    
#for sample in training_data[:10]:               #Check if data has Shuffled
    #print(sample[1])                                #in the first 10 itens

#Seperate the list by the features and labels
for features, labels in training_data:
    X_train_set.append(features)
    y_train_set.append(labels)

for features, labels in test_data:
    X_test_set.append(features)
    y_test_set.append(labels)

# X needs to be a np.array
X_train_set = np.array(X_train_set).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
X_test_set = np.array(X_test_set).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

y_train_set = np.array(y_train_set).reshape(1, -1)
y_test_set = np.array(y_test_set).reshape(1, -1)


save_training_data(X_train_set, y_train_set)
save_test_data(X_test_set, y_test_set) 









# Oppening image with Pillow 
#image_array = np.array(Image.open(r'G:\Meu Drive\Projects\Project Text Background\Images\Red\Red_1.jpg').resize((200, 200)))

#print(data)

# print(image.format)
# print(image.size)
# print(image.mode)

# image.show()

# Oppening image with matplotlib 
# image = image.imread(r'D:\User\Adriel\Projects\Project_Text_Background\Images\Red\Red_1.jpg')

# print(image.dtype)
# print(image.shape)

# pyplot.imshow(image)
# pyplot.show()

