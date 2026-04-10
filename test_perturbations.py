import os
from pathlib import Path
import cv2 # btw to install this do pip install opencv-python
import numpy as np
from deepface import DeepFace 
from deepface.models.demography import Gender 
# import tensorflow as tf
import tensorflow.compat.v1 as tf
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

def dataset_array(folder, limit=1000):
    images = []
    labels = []
    
    for i, (img, label) in enumerate(dataset_generator(folder)):
        if i >= limit:
            break
            
        images.append(img)
        labels.append(label)
    
    return np.array(images), np.array(labels)

# STEPS TWO + THREE: GET GRADIENT + FEEDFORWARD FUNCTIONS
# doing gender model at first bc its the simplest
tf.disable_v2_behavior()
print("Loading gender model from deepface...")
keras_model = Gender.load_model()
keras_model.trainable = False

# ok so UAP demo uses graphs
print("Dealing with graph mode...")
tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
y = tf.compat.v1.placeholder(tf.float32, shape=(None, 2))
logits = keras_model(x)

grads_list = []
num_classes = 2  # for gender

for i in range(num_classes):
    grad_i = tf.gradients(logits[:, i], x)[0]
    grads_list.append(grad_i)

grads_tensor = tf.stack(grads_list)  
# shape: (num_classes, batch, H, W, C)

sess = tf.compat.v1.Session()
print("Running session global variables intializer...")
sess.run(tf.compat.v1.global_variables_initializer())

# UAP wants the feedforward and gradient funcs now. so 
def f(x_np):
    return sess.run(logits, feed_dict={x: x_np})

def grads_f(x_np, class_indices):
    # ensure batch dimension
    if x_np.ndim == 3:
        x_np = x_np[None, ...]

    grads_out = sess.run(grads_tensor, feed_dict={x: x_np})

    # select only requested classes
    selected = grads_out[class_indices]

    return selected

test_dataset, test_labels = dataset_array("test_utk_dataset/")
v = np.load("data/universal.npy")

# test error
test_num_images = np.shape(test_dataset)[0]
batch_size = 100
num_batches = int(np.ceil(test_num_images / batch_size))
test_dataset_perturbed = v + test_dataset
test_est_labels_orig = np.zeros((test_num_images))
test_est_labels_pert = np.zeros((test_num_images))

for ii in range(0, num_batches):
  m = (ii * batch_size)
  M = min((ii+1)*batch_size, test_num_images)
  test_est_labels_orig[m:M] = np.argmax(f(test_dataset[m:M, :, :, :]), axis=1).flatten()
  test_est_labels_pert[m:M] = np.argmax(f(test_dataset_perturbed[m:M, :, :, :]), axis=1).flatten()

test_fooling_rate = float(np.sum(test_est_labels_pert != test_est_labels_orig) / float(test_num_images))
print('TEST FOOLING RATE = ', test_fooling_rate)
