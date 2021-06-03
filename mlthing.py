import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

#setup classifier
clf = svm.SVC(gamma=0.001, C=100)

x,y = digits.data[:-10], digits.target[:-10]#loads data and targets except last point to be used for testing
clf.fit(x,y)

print('Prediction:',clf.predict(digits.data[-1]))#calls for prediction

plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation="nearest")#shows the images we're predicting
plt.show()
