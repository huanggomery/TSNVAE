import numpy as np
import matplotlib.pyplot as plt


train_loss = np.load("save/train_loss.npy")
eval_loss = np.load("save/eval_loss.npy")

n = train_loss.shape[0]
x1 = np.arange(1, n+1, 1)
x2 = np.arange(1, n+1, 10)


plt.plot(x1, train_loss)
plt.plot(x2, eval_loss)
plt.show()