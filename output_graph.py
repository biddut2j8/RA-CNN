import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
history= pd.read_csv(f'./MD_20/2022-09-23_1642_1501/history.csv')
N = len(history)
plt.figure()

def network_one():
    plt.plot(np.arange(0, N), history["lambda_3_loss"], label="Training loss")
    plt.plot(np.arange(0, N), history["val_lambda_3_loss"], label="Validation loss")
    plt.title('Frequency Network Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.5)
    plt.legend(loc='upper right')
    plt.show()

def network_two():
    plt.plot(np.arange(0, N), history["add_7_loss"], label="Training loss")
    plt.plot(np.arange(0, N), history["val_add_7_loss"], label="Validation loss")
    plt.title('Image Network Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.5)
    plt.legend(loc='upper right')
    plt.show()

def loss():
    plt.plot(np.arange(0, N), history["loss"], label="Training loss")
    plt.plot(np.arange(0, N), history["val_loss"], label="Validation loss")
    plt.title('RA-Unet Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 0.5)
    plt.legend(loc='upper right')
    plt.show()

network_one()
network_two()
loss()
