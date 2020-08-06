import numpy as np
import scipy.io as io
import scipy.optimize as op
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def predict(theta1,theta2,X):
    m,n = X.shape

    X = np.append(np.ones([m,1]),X,axis=1)

    p = sigmoid(np.dot(np.append(np.ones([m,1]),sigmoid(np.dot(X,theta1.T)),axis=1),theta2.T))

    return np.argmax(p,axis=1)

input_layer_size  = 400 #20x20 pixels
hidden_layer_size = 25 #25 hidden numbers
num_labels = 10 #0-9

data = io.loadmat("images.mat")
X_data = data["X"]
y_data = data["y"]

fig,axis = plt.subplots(10,10,figsize=(8,8))
print("Visualizing Data:\n")

for i in range(10):
    for j in range(10):
        axis[i,j].imshow(X_data[np.random.randint(0,5000),:].reshape(20,20,order="F"),cmap='gray')
        axis[i,j].axis("off")

plt.draw()
plt.pause(4)
plt.close()

print("Loading Neural Network Parameters.\n")
weights = io.loadmat("ref-weights.mat")

theta1 = weights["Theta1"]
theta2 = weights["Theta2"]  

output = predict(theta1,theta2,X_data) + 1
pred = {1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:0}
print(f"Accuracy of the model: {np.mean(np.where(output==y_data.flatten(),1,0))*100}%\n")

print("View Examples: \n")

plt.figure()
while True:
    n = np.random.randint(0,5000)
    plt.imshow(X_data[n].reshape(20,20,order="F"),cmap='gray')
    plt.axis("off")
    plt.show(block=False)
    print(f"Prediction: {pred[output[n]]}")
    if input("Press enter to see another example or q to exit:\n") == 'q':
        break
    elif not plt.get_fignums():
        print("Graph window closed, ending program")
        break
    else:
        continue 
    
    