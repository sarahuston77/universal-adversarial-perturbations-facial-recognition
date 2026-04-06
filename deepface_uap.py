import os
from pathlib import Path
import cv2 # btw to install this do pip install opencv-python
import numpy as np
from deepface import DeepFace 
from deepface.models.demography import Gender 
import tensorflow as tf
from universal_pert import universal_perturbation

print("Done importing...")

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

def dataset_list(folder, limit=1000):
    data = []
    for i, (img, label) in enumerate(dataset_generator(folder)):
        if i >= limit:
            break
        data.append((img[None, ...], label[None, ...]))
    return data

# STEPS TWO + THREE: GET GRADIENT + FEEDFORWARD FUNCTIONS
# doing gender model at first bc its the simplest
print("Loading gender model from deepface...")
keras_model = Gender.load_model()
keras_model.trainable = False

# ok so UAP demo uses graphs
print("Dealing with graph mode...")
tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
logits = keras_model(x)
#! TODO: bug here^^^
# "RuntimeError: Exception encountered when calling layer 'conv2d' (type Conv2D). resource: 
# Attempting to capture an EagerTensor without building a function."
loss = tf.keras.losses.categorical_crossentropy(y, logits)
grad = tf.gradients(loss, x)[0]
sess = tf.compat.v1.Session()
print("Running session global variables intializer...")
sess.run(tf.compat.v1.global_variables_initializer())

# UAP wants the feedforward and gradient funcs now. so 
def f(x_np):
    return sess.run(logits, feed_dict={x: x_np})

def grad_f(x_np, y_np):
    return sess.run(grad, feed_dict={x: x_np, y: y_np})

# STEP FOUR: actually run perturbation 
print("Running UAP algorithm...")
v = universal_perturbation(
    dataset_list(dataset_generator(Path('data/crop_part1'))),
    f,
    grad_f,
    delta=0.2,
    max_iter_uni=10,
    xi=10/255.0,
    p=np.inf
)