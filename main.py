# import numpy as np
#
# a = np.array([2])
# print("a:", a)
# print("a size:", a.ndim)
#
# b = np.array([122, 127, 122], np.int8)
# b[0] = 122
# b[2] = 123
# print(b)
#
# print(np.multiply(a, b))
#
# c = np.diag([1,2,3])
# print("C:\n", c)
#
#
# d = np.array([[1, 2, 3], [4, 5, 6]])
# e = d.flatten()
#
# print(e)


# import pandas as pd
# import urllib
# import urllib.request
# url = "https://raw.githubusercontent.com/LisonEvf/practicalAI-cn/master/data/titanic.csv"
# request = urllib.request.urlopen(url)
# html = request.read()
# with open('data.csv', 'wb') as f:
#     f.write(html)
#
# df = pd.read_csv('data.csv', header=0)
#
# df.describe()
#
# print(df.describe())
#
# df["age"].hist()
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

tree = DecisionTreeRegressor()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree.fit(X_train, y_train)

y_pre = tree.predict(X_test)

accuracy = accuracy_score(y_test, y_pre)

print(accuracy)

#
# X, y = load_diabetes(return_X_y=True)
#
# X = X[:, [2]]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=40, shuffle=False)
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
#
# print(f"Mean squared error: {mean_squared_error(y_test, y_pred):.2f}")
# print(f"Coefficient of determination: {r2_score(y_test, y_pred):.2f}")
#
# fig, ax = plt.subplots(ncols=2, figsize=(10, 5), sharex=True, sharey=True)
# ax[0].scatter(X_train, y_train, label="Train data points")
# ax[0].plot(
#     X_train,
#     model.predict(X_train),
#     linewidth=3,
#     color="tab:orange",
#     label="Model predictions",
# )
# ax[0].set(xlabel="Feature", ylabel="Target", title="Train set")
# ax[0].legend()
#
# ax[1].scatter(X_test, y_test, label="Test data points")
# ax[1].plot(X_test, y_pred, linewidth=3, color="tab:orange", label="Model predictions")
# ax[1].set(xlabel="Feature", ylabel="Target", title="Test set")
# ax[1].legend()
#
# fig.suptitle("Linear Regression")
#
# plt.show()
