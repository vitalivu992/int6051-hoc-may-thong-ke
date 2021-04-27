from sklearn.svm import SVC

X = [[1, 9],
     [5, 5],
     [1, 1],
     [8, 5],
     [13, 1],
     [13, 9]]
y = [1, 1, 1, -1, -1, -1]
model = SVC(kernel='linear', C=1e5)
model.fit(X, y)
print("y = w^TX+b with \nw={}\nb={}".format(model.coef_, model.intercept_))
