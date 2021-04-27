import matplotlib.pyplot as plt
import numpy as np

X = np.array([[8.0, 8.5, 7.5]]).T
y = np.array([8.5, 9.5, 7.5])

one = np.ones((X.shape[0], 1))
Xbar = np.concatenate((one, X), axis=1)
XXT = np.dot(Xbar.T, Xbar)
XTY = np.dot(Xbar.T, y)
# %% md $w = [X X^T]^{-1} X^T Y$

# %%

W = np.dot(np.linalg.pinv(XXT), XTY)

w_0, w_1 = W[0], W[1]
print('y={w0:.2f}x+{w1:.2f}'.format(w0=w_0, w1=w_1))


def yhat(x: int):
    return w_0 + w_1 * x


plt.clf()
plt.plot(X, y, 'go', label='True data', alpha=0.5)
plt.legend(loc='best')
plt.show()
