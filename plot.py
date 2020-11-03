import numpy as np
import matplotlib.pyplot as plt

train_loss = np.load('train_loss_2fc_lr0.01.npy')
validation_loss = np.load('validation_loss_2fc_lr0.01.npy')

plt.plot(train_loss)
plt.plot(validation_loss)
plt.show()