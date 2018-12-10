import numpy as np

# define the lower and upper limits for x and y
minX, maxX, minY, maxY = 40, 80, 30, 70
# create one-dimensional arrays for x and y
lineSpacing = 10
x = np.linspace(minX, maxX, (maxX - minX)/lineSpacing + 1)
y = np.linspace(minY, maxY, (maxY - minY)/lineSpacing + 1)
# create the mesh based on these arrays
X, Y = np.meshgrid(x, y)
X = X.reshape((np.prod(X.shape),))
Y = Y.reshape((np.prod(Y.shape),))
coords = zip(X, Y)

for coord in coords:
    print(coord)
