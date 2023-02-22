import os
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
import numpy as np
import cv2



class OCR():
    def __init__(self, model_path):
        
        self.loaded_model = keras.models.load_model(model_path)
        print(self.loaded_model.summary())
        print('model loaded')
    def inferr(self, input_image):
        prediction = self.loaded_model.predict(input_image)
        return prediction
        


