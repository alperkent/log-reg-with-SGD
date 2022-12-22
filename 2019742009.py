import numpy as np
import matplotlib.pyplot as plt

# load data
testD = np.load('test_features.npy')
testL = np.load('test_labels.npy')
trainD = np.load('train_features.npy')
trainL = np.load('train_labels.npy')

trainN = len(trainL)
testN = len(testL)

# # scatterplot of data
# plt.scatter(testD[:264, 0], testD[:264, 1], facecolors='none', edgecolors='blue', marker='o')
# plt.scatter(testD[264:, 0], testD[264:, 1], c='red', marker='x')
# plt.title('Test Data')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.legend()
# plt.show()

# plt.scatter(trainD[:1005, 0], trainD[:1005, 1], facecolors='none', edgecolors='blue', marker='o')
# plt.scatter(trainD[1005:, 0], trainD[1005:, 1], c='red', marker='x')
# plt.title('Training Data')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.legend()
# plt.show()

# time-based learning schedule
def n_decay(n, k, d):
    return n/(1+d*k)

# logistic regression with gradient descent for 3 initial learning rate and 3 decay values
count_all = []
loss_all = []
w_all = []
n = [0.001, 0.01, 0.1]
d = [0.001, 0.01, 0.1]
y = trainL
x = np.hstack((np.ones((1561, 1)), trainD))
epochs = 100
for i in range(len(n)):
    for j in range(len(d)):
        w = np.zeros((3, 1))
        count = []
        loss = []
        for k in range(epochs):
            error = np.sum(np.log(1+np.e**(-y*np.dot(w.T, x.T))))/trainN
            count.append(k)
            loss.append(error)
            for t in range(trainN):
                ind = np.random.randint(0, trainN)
                g = -(y[ind]*x[ind])/(1+np.e**(y[ind]*np.dot(w.T, x[ind])))
                g.shape = (3, 1)
                w = w - n_decay(n[i], k, d[j])*g
        count_all.append(count)
        loss_all.append(loss)
        w_all.append(w)

# loss values per iteration for 3 eta and 3 decay
for i in range(len(n)*len(d)):
    plt.plot(count_all[i][:], loss_all[i][:], label='n='+str(n[i//3])+', d='+str(d[i%3]))
plt.title('Convergence curve')
plt.xlabel('Epoch count')
plt.ylabel('Loss values')
# plt.xscale('log')
# plt.yscale('log')
plt.legend()
plt.show()

# accuracy calculation
def accuracy(data, label, w):
    correct = 0
    for i in range(len(label)):
        result = np.sign(w[0]+(data[i, 0]*w[1])+(data[i, 1]*w[2]))
        if result == label[i]:
            correct += 1
    acc = (correct/len(label))*100
    return acc

print('Training accuracy:', accuracy(trainD, trainL, w_all[0]))
print('Test accuracy:', accuracy(testD, testL, w_all[0]))