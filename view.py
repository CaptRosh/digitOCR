import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

img = io.loadmat("images.mat")["X"]

fig,axis = plt.subplots(5,5,figsize=(8,8))

for i in range(5):
    for j in range(5):
        axis[i,j].imshow(img[np.random.randint(0,5000),:].reshape(20,20,order="F"),cmap = 'gray')
        axis[i,j].axis("off")

plt.show()