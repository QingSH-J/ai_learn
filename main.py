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
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.metrics import accuracy_score
#
# iris = load_iris()
# X = iris.data
# y = iris.target
#
# tree = DecisionTreeRegressor()
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# tree.fit(X_train, y_train)
#
# y_pre = tree.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pre)
#
# print(accuracy)

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
# import matplotlib.pyplot as plt
#
# from sklearn import svm
# from sklearn.datasets import make_blobs
# from sklearn.inspection import DecisionBoundaryDisplay
#
# # we create 40 separable points
# X, y = make_blobs(n_samples=40, centers=2, random_state=6)
#
# # fit the model, don't regularize for illustration purposes
# clf = svm.SVC(kernel="linear", C=1000)
# clf.fit(X, y)
#
# plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
#
# # plot the decision function
# ax = plt.gca()
# DecisionBoundaryDisplay.from_estimator(
#     clf,
#     X,
#     plot_method="contour",
#     colors="k",
#     levels=[-1, 0, 1],
#     alpha=0.5,
#     linestyles=["--", "-", "--"],
#     ax=ax,
# )
# # plot support vectors
# ax.scatter(
#     clf.support_vectors_[:, 0],
#     clf.support_vectors_[:, 1],
#     s=100,
#     linewidth=1,
#     facecolors="none",
#     edgecolors="k",
# )
# plt.show()
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

# Plot the results
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black",
            c="darkorange", label="data")
plt.plot(X_test, y_1, color="cornflowerblue",
         label="max_depth=2", linewidth=2)
plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()