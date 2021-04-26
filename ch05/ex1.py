import matplotlib.pyplot as plt
import numpy as np

X = np.array([[147, 150, 153, 158,
               163, 165, 168, 170, 173,
               175, 178, 180, 183]]).T
y = np.array([49, 50, 51, 54,
              58, 59, 60, 62, 63,
              64, 66, 67, 68])

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)

w_0, w_1 = w[0], w[1]
print('y={w0:.2f}x+{w1:.2f}'.format(w0=w_0, w1=w_1))


def yhat(x: int):
    return w_0 + w_1 * x


plt.clf()
plt.plot(X, y, 'go', label='True data', alpha=0.5)
plt.plot([155, 160], [yhat(155), yhat(160)], label='Predicted', alpha=0.5)
plt.legend(loc='best')
plt.show()
