import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2
import os
from scipy import ndimage
from numpy import asarray
from matplotlib.colors import hsv_to_rgb
from PIL import Image

number = 31 
#green = (90, size, 250)
#blue = (140, size, 225)
#red = (0, size, 225)

'''
for size in range(150, 250, 10):
    #color = (130, size, 220)
    color = (160, size, 150)
    img = np.full(((size+200)*2, (size+240)*2, 3), color, dtype=np.uint8) / 255.0
    plt.imshow(hsv_to_rgb(img))
    plt.axis('off')
    #plt.show()
    plt.savefig('G:\\Meu Drive\\Projects\\Project Text Background\\data\\train\\blue\\' + str(number), bbox_inches='tight', pad_inches = 0)
    number += 1
'''