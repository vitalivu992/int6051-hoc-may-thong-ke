import numpy as np
from sklearn.naive_bayes import MultinomialNB

# train data
# view | temp | humi | wind
# view: sunny = 0, overcast = 1, rain = 2
# temp: hot = 0, mild = 1, cool = 2
# humi: high = 0, normal = 1, low = 2
# wind: week = 0, strong = 1
train_data = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 0],
    [2, 1, 0, 0],
    [2, 2, 1, 0],

    [2, 2, 1, 1],
    [1, 2, 1, 1],
    [0, 1, 0, 0],
    [0, 2, 1, 0],
    [2, 1, 1, 1],

    [0, 1, 1, 1],
    [1, 1, 0, 0],
    [1, 0, 1, 0],
    [2, 1, 0, 1]
])
label = np.array(['N', 'N', 'Y', 'Y', 'Y',
                  'N', 'Y', 'N', 'Y', 'Y',
                  'Y', 'Y', 'Y', 'N'])

# test data

e4 = np.array([[2, 1, 2, 1]])

clf1 = MultinomialNB(alpha=1)

# training

clf1.fit(train_data, label)

# test

print('Probability of e4 in each class:', clf1.predict_proba(e4))
print('Predicting class of e4:', str(clf1.predict(e4)[0]))
