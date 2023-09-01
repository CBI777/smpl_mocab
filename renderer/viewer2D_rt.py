# Copyright (c) Facebook, Inc. and its affiliates.

#Visualization Function

import cv2

import numpy as np
import PIL
from PIL.Image import Image

def __ValidateNumpyImg(inputImg):
    if isinstance(inputImg, Image):
        # inputImg = cv2.cvtColor(np.array(inputImg), cv2.COLOR_RGB2BGR)
        inputImg = np.array(inputImg)

    return inputImg     #Q? is this copying someting (wasting memory or time?)?

def ImShow(inputImg, waitTime=1, bConvRGB2BGR=False,name='image', scale=1.0):

    inputImg = __ValidateNumpyImg(inputImg)

    if scale!=1.0:
        inputImg = cv2.resize(inputImg, (inputImg.shape[0]*int(scale), inputImg.shape[1]*int(scale)))


    if bConvRGB2BGR:
        inputImg = cv2.cvtColor(inputImg, cv2.COLOR_RGB2BGR)

    cv2.imshow(name, inputImg)
    key = cv2.waitKey(waitTime)
    # CBI :
    # Taking care of key inputs if bbox+rendered image window is shown.
    # 1. ESC : Shut down the program
    # 2. Spacebar : recalculate shape
    if key == 27:  # exit on ESC
        return 'Exit'
    if key == 32:
        print("Recalculating shape")
        return 'Shape'

    return None

