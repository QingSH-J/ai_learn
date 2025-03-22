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
# import numpy as np
# from sklearn.tree import DecisionTreeRegressor
# import matplotlib.pyplot as plt
#
# # Create a random dataset
# rng = np.random.RandomState(1)
# X = np.sort(5 * rng.rand(80, 1), axis=0)
# y = np.sin(X).ravel()
# y[::5] += 3 * (0.5 - rng.rand(16))
#
# # Fit regression model
# regr_1 = DecisionTreeRegressor(max_depth=2)
# regr_2 = DecisionTreeRegressor(max_depth=5)
# regr_1.fit(X, y)
# regr_2.fit(X, y)
#
# # Predict
# X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
# y_1 = regr_1.predict(X_test)
# y_2 = regr_2.predict(X_test)
#
# # Plot the results
# plt.figure()
# plt.scatter(X, y, s=20, edgecolor="black",
#             c="darkorange", label="data")
# plt.plot(X_test, y_1, color="cornflowerblue",
#          label="max_depth=2", linewidth=2)
# plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()

# import numpy as np
#
# a = np.array([1, 2, 3])
# b = a[:2].copy()
# b += 1
# print(a)
# print(b)

# import numpy as np
#
# c = np.arange(2, 10, 0.1)
# print(c)
#
# import numpy as np
#
# d = np.eye(5, dtype=int)
# print(d)

# import numpy as np
# e = np.diag([1, 2, 3, 4, 5], -2)
# print(e)
#
# try:
#     from scipy.datasets import face
# except ImportError:  # Data was in scipy.misc prior to scipy v1.10
#     from scipy.misc import face
#
# import matplotlib.pyplot as plt
#
# from numpy import linalg
# img = face()
#
#
# print(type(img))
#
# print(img.shape)
#
# img_array = img / 255
#
# img_grey = img_array @ [0.2, 0.7, 0.07]
#
# U, s, Vt = linalg.svd(img_grey)
#
# print(s.shape)
#
# Sigma = np.zeros((U.shape[1], Vt.shape[0]))
# np.fill_diagonal(Sigma, s)
#
# print(linalg.norm(img_grey - U @ Sigma @ Vt))
#
# if np.allclose(img_grey, U @ Sigma @ Vt):
#     print("True")
# else:
#     print("False")
# # plt.imshow(img_grey, cmap="grey")
# #
# # plt.show()
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
#
# # 1. 生成带有噪声的数据
# np.random.seed(42)  # 设置随机种子，确保结果可重现
# x = np.linspace(0, 10, 20)  # 20个x点，范围0到10
# true_slope = 2.5  # 真实斜率
# true_intercept = 1.0  # 真实截距
# y_true = true_slope * x + true_intercept  # 真实直线
# noise = np.random.normal(0, 1.5, size=len(x))  # 添加高斯噪声
# y = y_true + noise  # 带噪声的观测值
#
# # 2. 手动实现最小二乘法
# # 公式: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - (sum(x))^2)
# # 公式: intercept = (sum(y) - slope*sum(x)) / n
# n = len(x)
# sum_x = np.sum(x)
# sum_y = np.sum(y)
# sum_xy = np.sum(x * y)
# sum_x2 = np.sum(x**2)
#
# slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
# intercept = (sum_y - slope * sum_x) / n
#
# # 3. 使用NumPy的polyfit函数（更简洁的方法）
# slope_np, intercept_np = np.polyfit(x, y, 1)
#
# # 4. 计算预测值
# y_pred = slope * x + intercept
#
# # 5. 计算均方误差(MSE)
# mse = np.mean((y - y_pred)**2)
#
# # 6. 可视化结果
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, color='blue', label='数据点')
# plt.plot(x, y_true, 'g-', label=f'真实模型: y = {true_slope}x + {true_intercept}')
# plt.plot(x, y_pred, 'r--', label=f'拟合结果: y = {slope:.3f}x + {intercept:.3f}')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('最小二乘法线性拟合示例')
# plt.legend()
# plt.grid(True)
# plt.text(1, 25, f'均方误差(MSE): {mse:.3f}')
# plt.show()
#
# print(f"手动计算: 斜率 = {slope:.4f}, 截距 = {intercept:.4f}")
# print(f"NumPy计算: 斜率 = {slope_np:.4f}, 截距 = {intercept_np:.4f}")

# import pandas as pd
# import numpy as np
# s = pd.Series([1, 2, 3, 4, 5], dtype=float)
#
# dates = pd.date_range("1/1/2025", "1/6/2025")
#
# df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
# print(df)

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 设置随机种子保证结果可重复
# torch.manual_seed(42)
#
#
# # 创建二次函数 f(x) = ax² + bx + c
# def quadratic_function(x, a, b, c):
#     return a * x ** 2 + b * x + c
#
#
# # 定义参数
# a, b, c = 2.0, -3.0, 5.0
#
# # 创建需要进行微分的点（需要设置requires_grad=True）
# x_points = torch.linspace(-3, 3, 100, requires_grad=True)
#
# # 计算函数值
# y = quadratic_function(x_points, a, b, c)
#
# # 创建一个图表显示原始函数
# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.plot(x_points.detach().numpy(), y.detach().numpy(), 'b-', linewidth=2)
# plt.title(f'二次函数: f(x) = {a}x² + {b}x + {c}')
# plt.grid(True)
# plt.ylabel('f(x)')
#
# # 计算每个点的导数
# derivatives = []
# analytical_derivatives = []  # 解析导数用于比较
#
# for i in range(len(x_points)):
#     # 每次迭代需要清除之前的梯度
#     if x_points.grad is not None:
#         x_points.grad.zero_()
#
#     # 选择单个点计算
#     x_single = x_points[i]
#     y_single = quadratic_function(x_single, a, b, c)
#
#     # 反向传播计算梯度
#     y_single.backward(retain_graph=True)
#
#     # 保存计算的导数
#     derivatives.append(x_points.grad[i].item())
#
#     # 计算解析导数: f'(x) = 2ax + b
#     analytical_derivatives.append(2 * a * x_single.item() + b)
#
# # 显示导数
# plt.subplot(2, 1, 2)
# plt.plot(x_points.detach().numpy(), derivatives, 'r-', linewidth=2, label='自动微分')
# plt.plot(x_points.detach().numpy(), analytical_derivatives, 'g--', linewidth=2, label='解析公式: 2ax + b')
# plt.title('函数的导数')
# plt.grid(True)
# plt.ylabel("f'(x)")
# plt.xlabel('x')
# plt.legend()
#
# # 输出特定点的导数值
# test_points = [-2.0, -1.0, 0.0, 1.0, 2.0]
# print("\n特定点的导数值比较:")
# print("    x    |  自动微分  |  解析公式  |  差异")
# print("---------+------------+------------+--------")
#
# for point in test_points:
#     # 创建一个需要求导的点
#     x_test = torch.tensor(point, requires_grad=True, dtype=torch.float32)
#     y_test = quadratic_function(x_test, a, b, c)
#     y_test.backward()
#
#     autograd_derivative = x_test.grad.item()
#     analytical_derivative = 2 * a * point + b
#     difference = abs(autograd_derivative - analytical_derivative)
#
#     print(f"{point:8.2f} | {autograd_derivative:10.6f} | {analytical_derivative:10.6f} | {difference:8.2e}")
#
# plt.tight_layout()
# plt.show()

# import torch
#
#
# def f(x, y):
#     return x ** 2 + y ** 4 + 2
#
#
# def calculate():
#     n = 10000  # 增加迭代次数
#     alpha = 0.1  # 增加学习率
#     x = torch.tensor([1.1], requires_grad=True)
#     y = torch.tensor([2.2], requires_grad=True)
#
#     print(f'起始点: x = {x.item():.6f}, y = {y.item():.6f}, f(x,y) = {f(x, y).item():.6f}')
#
#     for i in range(1, n + 1):
#         z = f(x, y)
#         z.backward()
#         x.data -= alpha * x.grad.data
#         y.data -= alpha * y.grad.data
#
#         x.grad.zero_()
#         y.grad.zero_()
#
#         print(f'迭代 {i}: x = {x.item():.6f}, y = {y.item():.6f}, f(x,y) = {f(x, y).item():.6f}')
#
#     print(f'最终结果: x = {x.item():.6f}, y = {y.item():.6f}, f(x,y) = {f(x, y).item():.6f}')
#
#
# calculate()


import torch

def f(x):
    return 2 * x ** 2 + 3 * x - 4;

def calculate():
    x = torch.tensor([1.1], requires_grad=True)
    y = torch.tensor([2.1], requires_grad=True)
    n = 1000
    alpha = 0.1
    for i in range(1, n + 1):
        z = f(x)
        z.backward()
        x.data -= alpha * x.grad.data
        x.grad.zero_()
        print(f'迭代 {i}: x = {x.item():.6f}, f(x) = {f(x).item():.6f}')

calculate()