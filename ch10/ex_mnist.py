import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
import numpy as np
data_dir = 'data'
mnist = fetch_openml('mnist_784', version=1, data_home=data_dir, as_frame=False)

print("Shape of mnist", mnist.data.shape)
k = 10
N = 10000

X = mnist.data[np.random.choice(mnist.data.shape[0], N)]
kmeans = KMeans(n_clusters=k).fit(X)
predict = kmeans.predict(X)