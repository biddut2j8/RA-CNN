

'''
import numpy as np
from PIL import Image
from keras.datasets import mnist

def img_show(img):
    pil_image = Image.fromarray(np.uint8(img))
    image_arr = np.array(pil_image)
    image_arr = 255 - image_arr
    pil_image = Image.fromarray(image_arr)
    pil_image.show()

(x_train, t_train), (x_test, t_test) = mnist.load_data()
img = (x_train[0])
img = img.reshape(28, 28)
img_show(img)

'''

import sys, os
sys.path.append(os.pardir)  
import numpy as np
import pickle
from keras.datasets import mnist
from keras.activations import sigmoid, softmax

def get_data():
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # Get the index of the element with the highest probability.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
