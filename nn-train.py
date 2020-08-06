import numpy as np
import scipy.io as io
import scipy.optimize as op
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def randInit(l_in,l_out):
    eps = 0.12
    return np.random.rand(l_out,1+l_in) * (2*eps) - eps

def nnCostFunction(nn_params, input_layer_size,hidden_layer_size,num_labels,X,y,lambda_val):
    theta1 = nn_params[:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels,hidden_layer_size+1)
    
    row = X.shape[0]

    #Converting Y from integer to a set of 1s and 0s
    y_new = np.zeros([num_labels, row])
    for i in range(row):
        y_new[y[i],i] = 1

    # Forward Propagation
    a1 = np.append(np.ones([row,1]),X,axis=1)
    a2 = sigmoid(np.dot(theta1,a1.T))
    a2 = np.append(np.ones([row,1]), a2.T,axis=1)
    a3 = sigmoid(np.dot(theta2, a2.T))
    h_theta = a3
    
    J = -1/row * np.sum(np.sum((y_new * np.log(h_theta)) + ((1-y_new) * np.log(1 - h_theta))))

    reg = lambda_val/(2*row) * (np.sum(np.sum(theta1[:,1:]**2)) + np.sum(np.sum(theta2[:,1:]**2)))

    return J + reg

def nnGradFunction(nn_params, input_layer_size,hidden_layer_size,num_labels,X,y,lambda_val):
    theta1 = nn_params[:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].reshape(num_labels,hidden_layer_size+1)
    
    row = X.shape[0]

    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)
    
    y_new = np.zeros([num_labels, row])
    for i in range(row):
        y_new[y[i],i] = 1

    #Back-Propagation
    for i in range(row):
        a1 = np.append(1,X[i,:])
        a2 = sigmoid(np.dot(theta1, a1))
        a2 = np.append(1, a2)
        a3 = sigmoid(np.dot(theta2,a2))

        delta3 = a3 - y_new[:,i]
        delta2 = np.dot(theta2.T,delta3) * sigmoidGradient(np.append(1,np.dot(theta1,a1)))

        delta2 = delta2[1:] 

        theta2_grad += delta3.reshape([10,1]) * a2.reshape([1,26])
        theta1_grad += delta2.reshape([25,1]) * a1.reshape([1,401])

    theta1_grad *= 1/row
    theta2_grad *= 1/row

    theta1_grad[:,1:] += lambda_val/row * theta1[:,1:]
    theta2_grad[:,1:] += lambda_val/row * theta2[:,1:]

    grad = np.append(theta1_grad.flatten(),theta2_grad.flatten())

    return grad

def predict(theta1,theta2,X):
    row = X.shape[0]
    num_labels = theta2.shape[0]

    X = np.append(np.ones([row,1]),X,axis=1)
    h1 = sigmoid(np.dot(X , theta1.T))

    h1 = np.append(np.ones([row,1]),h1,axis=1)
    h2 = sigmoid(np.dot(h1,theta2.T))
    
    return np.argmax(h2,axis=1).reshape(row,1)

input_layer = 400
hidden_layer = 25
num_labels = 10

print("Loading and Visualizing data...\n")

data = io.loadmat("images.mat")
X_data = data['X']
y_data = data['y'] - 1 #To adjust for 0s mapping to 10


row,col = X_data.shape

fig,axis = plt.subplots(10,10,figsize=(8,8))

for i in range(10):
    for j in range(10):
        axis[i,j].imshow(X_data[np.random.randint(0,5000),:].reshape(20,20,order="F"))
        axis[i,j].axis("off")

plt.draw()
plt.pause(1.5)
plt.close()

print(f"Loading Weights...\n")
weights = io.loadmat("weights.mat")
theta1 = weights["Theta1"]
theta2 = weights["Theta2"]

nn_params = np.append(theta1.flatten(), theta2.flatten())

init_lambda = 0

init_J = nnCostFunction(nn_params,input_layer,hidden_layer,num_labels,X_data,y_data,init_lambda)
init_Grad = nnGradFunction(nn_params,input_layer,hidden_layer,num_labels,X_data,y_data,init_lambda)

print(f"Cost function with lambda = 0 (from ex4weights): {init_J:0.6f}")
print(f"This value should be about 0.287629\n")

reg_lambda = 1

reg_J = nnCostFunction(nn_params,input_layer,hidden_layer,num_labels,X_data,y_data,reg_lambda)
reg_Grad = nnGradFunction(nn_params,input_layer,hidden_layer,num_labels,X_data,y_data,reg_lambda)

print(f"Cost function with lambda = 1 (from ex4weights): {reg_J:0.6f}")
print(f"This value should be: 0.383770\n")

g = sigmoidGradient(np.array([-1 ,-0.5, 0 ,0.5 ,1]))

print(f"""Sigmoid Gradient for [-1 ,-0.5, 0 ,0.5 ,1]:
{g[0]:.6}
{g[1]:.6}
{g[2]:.6}
{g[3]:.6}
{g[4]:.6}\n""")
print(f"""These values should be: 
0.196612
0.235004
0.250000
0.235004
0.196612\n""")

test_theta1 = randInit(input_layer,hidden_layer)
test_theta2 = randInit(hidden_layer,num_labels)

print(f"""Gradient before regularization(first six values):
{init_Grad[0]:.6}
{init_Grad[1]:.6}
{init_Grad[2]:.6}
{init_Grad[3]:.6}
{init_Grad[4]:.6}
{init_Grad[5]:.6}\n""")

print(f"""These values should be:
-0.0092782524
 0.0088991196
-0.0083601076
0.0076281355
-0.0067479837
-0.0000030498\n""")

print(f"""Gradient after regularization(first six values):
{reg_Grad[0]:.6}
{reg_Grad[1]:.6}
{reg_Grad[2]:.6}
{reg_Grad[3]:.6}
{reg_Grad[4]:.6}
{reg_Grad[5]:.6}\n""")

print(f"""These values should be:
-0.009278252
0.008899120
-0.008360108
0.007628136
-0.006747984
-0.016767980\n""")

test_params = np.append(test_theta1.flatten(),test_theta2.flatten())
test_lambda = 3

test_J = nnCostFunction(nn_params, input_layer, hidden_layer, num_labels, X_data, y_data, test_lambda)
test_Grad = nnGradFunction(nn_params, input_layer, hidden_layer, num_labels, X_data, y_data, test_lambda)

print(f"Cost at debugging parameters(lambda = 3): {test_J:0.6f}")
print("This value should be: 0.576051\n")

print("Training Neural Network...\n")

res_params = op.fmin_cg(f=nnCostFunction,fprime=nnGradFunction,x0=test_params, args=(input_layer, hidden_layer, num_labels, X_data, y_data, test_lambda), maxiter=50)

test_theta1 = res_params[:hidden_layer*(input_layer+1)].reshape(hidden_layer,input_layer+1)
test_theta2 = res_params[hidden_layer*(input_layer+1):].reshape(num_labels,hidden_layer+1)

pred = predict(test_theta1,test_theta2,X_data)

accuracy = np.mean(np.where(pred == y_data,1,0))

fig,axis = plt.subplots(1,1,figsize=(8,8))

print(f"\nVisualizing Neural Network...\n")
# iter = 0
# for i in range(5):
#     for j in range(2):
#         axis[i,j].imshow(X_data[np.random.randint(500*iter,500*(iter+1)),:].reshape(20,20,order="F"))
#         axis[i,j].axis("off")
#         iter += 1

# plt.draw()

test_data = X_data[np.random.randint(0,10000),:]
ans = predict(test_theta1,test_theta2,test_data)
axis[0,0].imshow(test_data.reshape(20,20,order="F"))
print(ans)  

print(f"The accuracy of the neural network model is: {accuracy*100:0.2f}%")

plt.show()