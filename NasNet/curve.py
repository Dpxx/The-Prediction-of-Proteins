import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.log')
epoch = data['epoch']
acc = data['acc']
loss = data['loss']
val_acc = data['val_acc']
val_loss = data['val_loss']

plt.figure()

plt.subplot(2, 1, 1)
plt.plot(epoch, loss, label='train_loss')
plt.plot(epoch, val_loss, label='val_loss')
plt.legend()
plt.title('loss')

plt.subplot(2, 1, 2)
plt.plot(epoch, acc, label='train_acc')
plt.plot(epoch, val_acc, label='val_acc')
plt.legend()
plt.title('acc')

plt.show()