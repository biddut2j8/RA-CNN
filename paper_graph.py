import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
history= pd.read_csv(f'./MD_25/2022-09-25_1705_1501/history.csv')
N = len(history)
plt.figure()
#figsize=(9,8), dpi = 300
plt.rcParams['font.size'] = '10'
def network_one():
    plt.plot(np.arange(0, N), history["lambda_3_loss"], label="Training loss", linewidth= 1)
    plt.plot(np.arange(0, N), history["val_lambda_3_loss"], label="Validation loss", linewidth= 1)
    #plt.title('Frequency Network Loss', fontsize=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.xticks([0, 500, 1000, 1500])
    plt.ylim(0, 0.5)
    #plt.yticks([0, 0.15, 0.30, 0.45])
    plt.legend(loc='upper right')
    plt.show()

def network_two():
    plt.plot(np.arange(0, N), history["add_7_loss"], label="Training loss")
    plt.plot(np.arange(0, N), history["val_add_7_loss"], label="Validation loss")
    #plt.title('Image Network Loss', fontsize=10)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks([0, 500, 1000, 1500])
    plt.ylim(0, 0.5)
    #plt.yticks([0, 0.15, 0.30, 0.45])
    plt.legend(loc='upper right')
    plt.show()

def loss():
    plt.plot(np.arange(0, N), history["loss"], label="Training loss")
    plt.plot(np.arange(0, N), history["val_loss"], label="Validation loss")
    #plt.title('RA-Unet Model Loss', fontsize=10)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks([0, 500, 1000, 1500])
    plt.ylim(0, 0.30)
    #plt.yticks([0, 0.15, 0.30])
    plt.legend(loc='best')
    plt.show()

#network_one()
#network_two()
loss()

#plt.plot(np.arange(0, N), history["loss"], label="Training loss", linewidth=1)
#plt.plot(np.arange(0, N), history["val_loss"], label="Validation loss", linewidth=1)