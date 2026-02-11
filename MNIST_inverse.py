# -*- coding: utf-8 -*-
"""
To use the trained model for inference and visualization, please refer to Model_demo.py

"""

import copy
import numpy as np
import torch
import tqdm
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchmetrics.regression import MeanAbsolutePercentageError
from google.colab import drive
drive.mount('/content/drive')

#-------load data
y_train=np.loadtxt('/content/drive/MyDrive/Colab Notebooks/MNIST_input_files/mnist_img_train.txt')
y_test=np.loadtxt('/content/drive/MyDrive/Colab Notebooks/MNIST_input_files/mnist_img_test.txt')
X_train1=np.loadtxt('/content/drive/MyDrive/Colab Notebooks/FEA_displacement_results_step12/summary_dispx_train_step12.txt')
X_test1=np.loadtxt('/content/drive/MyDrive/Colab Notebooks/FEA_displacement_results_step12/summary_dispx_test_step12.txt')
X_train2=np.loadtxt('/content/drive/MyDrive/Colab Notebooks/FEA_displacement_results_step12/summary_dispy_train_step12.txt')
X_test2=np.loadtxt('/content/drive/MyDrive/Colab Notebooks/FEA_displacement_results_step12/summary_dispy_test_step12.txt')

# --- normalize each part and compose input---
eps = 1e-8

m1 = X_train1.mean(axis=0, keepdims=True)
s1 = X_train1.std(axis=0, keepdims=True)
s1[s1 < eps] = 1.0

m2 = X_train2.mean(axis=0, keepdims=True)
s2 = X_train2.std(axis=0, keepdims=True)
s2[s2 < eps] = 1.0

X_train1 = (X_train1 - m1) / s1
X_test1  = (X_test1  - m1) / s1
X_train2 = (X_train2 - m2) / s2
X_test2  = (X_test2  - m2) / s2
X_train=np.hstack((X_train1,X_train2))
X_test=np.hstack((X_test1,X_test2))

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(784*2, 2048),
    nn.ReLU(),
    nn.Linear(2048,1024),
    nn.ReLU(),
    nn.Linear(1024, 784)
)


# -----loss function and optimizer
loss_fn = nn.L1Loss()# mean absolute error
optimizer = optim.Adam(model.parameters(), lr=0.0001)

n_epochs = 30   # number of epochs to run
batch_size = 256  # size of each batch
batch_start = torch.arange(0, len(X_train), batch_size)
best_mse = np.inf   # init to infinity
best_weights = None
history = []

loss_fn(y_test[1], y_test[2])


##-------train
for epoch in range(n_epochs):
    model.train()
    with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
        bar.set_description(f"Epoch {epoch}")
        for start in bar:
            # take a batch
            X_batch = X_train[start:start+batch_size]
            y_batch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            bar.set_postfix(mse=float(loss))
    # evaluate accuracy at end of each epoch
    model.eval()
    y_pred = model(X_test)
    mae = loss_fn(y_pred, y_test)
    mae = float(mae)
    history.append(mae)
    if mae < best_mse:
        best_mse = mae
        best_weights = copy.deepcopy(model.state_dict())
    print(mae)
model.load_state_dict(best_weights)


#-----test and visualization
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test).cpu().numpy()
    y_test_true = y_test.cpu().numpy() if torch.is_tensor(y_test) else np.asarray(y_test)

idx =np.random.randint(len(y_test_true))

true_img = y_test_true[idx].reshape(28, 28)
pred_img = y_test_pred[idx].reshape(28, 28)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(true_img, cmap="viridis")
plt.title("True")
plt.colorbar()
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pred_img, cmap="viridis")
plt.title("Predicted")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
plt.show()

# --------save trained model
save_path = "/content/drive/MyDrive/Colab Notebooks/inverse_fnn_checkpoint.pt"

checkpoint = {
    "model_state_dict": model.state_dict(),

    "X1_mean": m1,
    "X1_std":  s1,
    "X2_mean": m2,
    "X2_std":  s2,

    "input_dim": 1568,
    "output_dim": 784,
    "model_type": "FNN",
}

torch.save(checkpoint, save_path)
print(f"Model saved to {save_path}")

