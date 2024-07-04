# https://www.youtube.com/watch?v=YAJ5XBwlN4o

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# datos demo
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

y = y.view(y.shape[0], 1)

n_sample, n_features = X.shape

# definición del modelo
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# función de costo y optimizador
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# loop de entrenamiento

num_epocas = 1000
for epoca in range(num_epocas):
    # forward pass
    y_predicted = model(X)
    loss = criterion(y_predicted, y)

    # backward pass
    loss.backward()

    # update
    optimizer.step()
    optimizer.zero_grad()

    if (epoca + 1) % 10 == 0:
        print(f'epoca: {epoca}, loss = {loss.item():.4}')

# graficado
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
