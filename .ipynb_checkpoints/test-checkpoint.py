import pandas as pd
import numpy as np
data = pd.read_csv("iris_data.csv")
data= data.drop(['Unnamed: 0', 'Id'], axis=1)
X=data.drop(['Species'], axis=1)
y = data['Species']
X=X.values
y= y.values
y_2d = y[:, None]
Y = np.hstack([y_2d==i for i in range(3)]).astype(int)
n, d = X.shape  # number of data samples and features
k = 10  # number of hidden neurons
d = X.shape[1]  # number of data features
W = np.random.randn(d, k)

H = np.tanh(X@W)
H_bias = np.hstack([H, np.ones((n, 1))])
B_bias = np.linalg.pinv(H_bias)@Y

B, bias = B_bias[:-1], B_bias[-1]

Y_raw = np.tanh(X@W) @ B + bias
y_pred = Y_raw.argmax(axis=1)

accuracy = np.mean(y_pred == y)
print("Accuracy: {:.2f}%".format(100 * accuracy))