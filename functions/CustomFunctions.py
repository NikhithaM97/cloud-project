import tensorflow as tf
import numpy as np
from keras_preprocessing.image import img_to_array

def load_image(filename):
    img = img_to_array(filename)
    img = img.reshape(1, 32, 32, 3)
    img = img.astype('float32')
    img = img / 255.0
    return img

#Load the image
def Cifar10_Prediction(img, weights, classes, target_size):
 
    model = tf.keras.models.load_model(weights)   
    if img.mode != "RGB":
        img = img.convert("RGB")
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            img = img.resize(width_height_tuple)

    img = load_image(img)
    
    y_pred = model.predict(img)
    y_pred_classes = np.argmax(y_pred, axis = 1)
    classification = classes[y_pred_classes[0]]

    return classification

