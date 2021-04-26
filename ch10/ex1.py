import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
np.random.seed(18)

means = [[2, 2],
         [8, 3],
         [3, 6]]
cov = [[1, 0],
       [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
# plt.scatter(X0[:,0],X0[:,1], label='True Position')
# plt.show()
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis=0)

k = 3
model = KMeans(n_clusters=k,random_state=0).fit(X)
print("with K={}, centers are \n{}".format(k, model.cluster_centers_))
predict = model.predict(X)

plt.scatter(X[:,0],X[:,1], c=model.labels_, cmap='rainbow')
plt.show()