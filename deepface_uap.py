import os
import cv2 # btw to install this do pip install opencv-python
import numpy as np
import tensorflow as tf

# STEP ONE: LOAD DATA
IMG_SIZE = 224 
# deepface's gender/race/age model code indicates that images gotta be squares this big:
# Single image as np.ndarray (224, 224, 3)

def parse_label(filename):
    # [age] is an integer from 0 to 116, indicating the age
    # [gender] is either 0 (male) or 1 (female)
    # [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern)
    age, gender, race, _ = filename.split('_', 3)
    return int(age), int(gender), int(race)

def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return img

def dataset_generator(folder):
    for fname in os.listdir(folder):
        if not fname.endswith(".jpg"):
            continue
            
        path = os.path.join(folder, fname)
        age, gender, race = parse_label(fname)
        
        img = load_image(path)
        # the below is for gender; will change if age or race
        label = tf.keras.utils.to_categorical(gender, 2)
        
        yield img, label

# STEPS TWO + THREE: GET GRADIENT + FEEDFORWARD FUNCTIONS