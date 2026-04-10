from IPython.display import Image
from IPython.display import display
import os
from pathlib import Path
import cv2 # btw to install this do pip install opencv-python
import numpy as np
from deepface import DeepFace 
from deepface.models.demography import Gender 
import tensorflow as tf

print("Done importing...")

# STEP ONE: LOAD DATA
IMG_SIZE = 224 
# deepface's gender/race/age model code indicates that images gotta be squares this big:
# Single image as np.ndarray (224, 224, 3)

def parse_label(filename):
  # [age] is an integer from 0 to 116, indicating the age
  # [gender] is either 0 (male) or 1 (female)
  # [race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern)
  split = filename.split('_', 3)
  print(split)
  if len(split) != 4:
      age, gender, race = split[0], split[1], -1 # missing race, probably should delete these
  else:
      age, gender, race, _ = filename.split('_', 3)
      
  return int(age), int(gender), int(race)

def load_image(path):
  img = cv2.imread(path)
  img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
  img = img.astype("float32") / 255.0
  return img

def dataset_generator(name):   
  path = name
  age, gender, race = parse_label(name)
  
  img = load_image(path)
  # the below is for gender; will change if age or race
  label = tf.keras.utils.to_categorical(gender, 2)
  
  return np.array(img), np.array(label)

print("REGULAR")
display(Image(url="test_utk_dataset/100_1_2_20170105174847679.jpg.chip.jpg", width=300, height=300))
print("PERTURBATION")

img, label = dataset_generator("test_utk_dataset/100_1_2_20170105174847679.jpg.chip.jpg")
v = np.load("data/universal.npy")
perturbed = img + v
perturbed_bgr = (perturbed * 255).astype(np.uint8)
cv2.imwrite("temp_perturbed.jpg", perturbed_bgr)
display(Image(url="temp_perturbed.jpg", width=300, height=300))