import pandas as pd
import ast
import matplotlib.pyplot as plt
import numpy as np
dataframe = pd.read_excel('CVSHIT.xlsx',header=None)

NUM_X_POINTS = 11
NUM_Y_POINTS = 6
PIXEL_SIZE = 0.12

def flatten(to_flatten):
    flattened_list = []
    for sublist in to_flatten:
        for item in sublist:
            flattened_list.append(item)
    return flattened_list

dataframe = dataframe.applymap(lambda x: ast.literal_eval(x))
dataframe = dataframe.applymap(lambda x: (x[0]*PIXEL_SIZE,x[1]*PIXEL_SIZE))


flattened_list = flatten(dataframe.values)


print(flattened_list)
flattened_list = np.asarray(flattened_list)
plt.figure(0)
plt.scatter(flattened_list[:,0],flattened_list[:,1],c='red')



x = np.linspace(155*PIXEL_SIZE,490*PIXEL_SIZE,NUM_X_POINTS)
y = np.linspace(60*PIXEL_SIZE,240*PIXEL_SIZE,NUM_Y_POINTS)
X,Y = np.meshgrid(x,y)
print("X",X)
plt.scatter(X,Y,c='blue')

X = flatten(X)
Y = flatten(Y)

truth_flattened_list = []
for i in range(len(X)):
    truth_flattened_list.append((X[i],Y[i]))

errors = []
for i in range(len(truth_flattened_list)):
    truth = truth_flattened_list[i]
    pred = flattened_list[i]

    error = ((truth[0] - pred[0])**2 +   (truth[1] - pred[1])**2)**0.5
    print(truth,pred,error)
    errors.append(error)

print("errors",errors)
plt.figure(1)
plt.hist(errors)


plt.show()
