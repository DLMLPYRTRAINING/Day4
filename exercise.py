#
# # http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html
# # # -------------------------------------------------------------
# # # regression with all the other types of regression models
# # import
# from sklearn import linear_model
# from sklearn import svm
# from sklearn.metrics import r2_score,mean_squared_error
#
# classifiers1 = [
#     linear_model.LinearRegression()]
#
# classifiers2 = [
#     svm.SVR(),
#     linear_model.SGDRegressor(),
#     linear_model.BayesianRidge(),
#     linear_model.LassoLars(),
#     linear_model.ARDRegression(),
#     linear_model.PassiveAggressiveRegressor(),
#     linear_model.TheilSenRegressor()
# ]
#
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import time
#
# # import dataset
# data = pd.read_csv(r"https://raw.githubusercontent.com/DLMLPYRTRAINING/Day3/master/Datasets/Logistic_regression1.csv")
#
# #Example: various way of fetching data from pandas
# X = data.loc[0:len(data)-2, "Height"]
# print(X)
# X = data["Height"][:-1]
# print(X)
# X = data[:-1]["Height"]
# print(X)
#
# weight_val = [np.log(i/(1-i)) for i in data["Weight"].values]
#
# # fetch the data
# Train_F = data[:-1]["Height"].values.reshape(-1, 1)
# print(Train_F)
# plt.plot(Train_F, "b.")
# Train_T = data[:-1]["Weight"].values.reshape(-1, 1)
# print(Train_T)
# plt.plot(Train_T, "bo")
#
# Test_F = data[-1:]["Height"].values.reshape(-1, 1)
# print(Test_F)
# plt.plot(len(Train_F), Test_F, "g.")
# Test_T = data[-1:]["Weight"].values.reshape(-1, 1)
# print(Test_T)
# plt.plot(len(Train_T), Test_T, "go")
#
# for item in classifiers1:
#     # build model
#     model = item
#
#     # train model
#     model.fit(Train_F, Train_T)
#
#     # score model
#     score = model.score(Train_F, Train_T)
#
#     # predict
#     predict = model.predict(Test_F)
#
#     #error%
#     mse = mean_squared_error(Test_T,predict)
#     r2 = r2_score(Test_T,predict)
#
#     # print everything
#     print(item)
#     print("score\n", score)
#     print("predict:\n", predict)
#     print("actual:\n", Test_T)
#     print("mean_squared_error:\n",mse)
#     print("R2:\n",r2)
#     time.sleep(5)
#
# for item in classifiers2:
#     # build model
#     model = item
#
#     # train model
#     model.fit(Train_F,Train_T.ravel())
#     print(model.support_vectors_)
#
#     # score model
#     score = model.score(Train_F, Train_T.ravel())
#
#     # predict
#     predict = model.predict(Test_F)
#
#     #error%
#     mse = mean_squared_error(Test_T.ravel(),predict)
#     r2 = r2_score(Test_T.ravel(),predict)
#
#     # print everything
#     print(item)
#     print("score\n", score)
#     print("predict:\n", predict)
#     print("actual:\n", Test_T)
#     print("mean_squared_error:\n",mse)
#     print("R2:\n",r2)
#
#
# #show plot
# plt.show()

# ---------------------------------------------------------------
# # classification with logistic regression
# from sklearn.linear_model import LogisticRegression
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# data = pd.read_csv("https://raw.githubusercontent.com/DLMLPYRTRAINING/Day3/master/Datasets/Logistic_classification1.csv")
#
# #first we'll have to convert the strings "No" and "Yes" to numeric values
# data.loc[data["default"] == "No", "default"] = 0
# data.loc[data["default"] == "Yes", "default"] = 1
# X = data["balance"][:-1].values.reshape(-1, 1)
# Y = data["default"][:-1].values.reshape(-1, 1)
#
# x_test = data["balance"][-1:].values.reshape(-1, 1)
# y_test = data["default"][-1:].values.reshape(-1, 1)
#
# LogR = LogisticRegression()
# LogR.fit(X, np.ravel(Y.astype(int)))
# score = LogR.score(X, np.ravel(Y.astype(int)))
# print("Score:\n", score)
#
# coeff = LogR.coef_
# intercept = LogR.intercept_
# print("Coeff\n", coeff)
# print("Intercept\n", intercept)
#
# predict = LogR.predict(x_test)
# predict_proba_class = LogR.predict_proba(x_test)
# print("Prediction Feature:\n", x_test)
# print("Prediction Value:\n", predict)
# print("Actual Value:\n",y_test)
# print("Prediction Class:\n", predict_proba_class)
#
# def model_plot(x):
#     return 1/(1+np.exp(-x))
#
# points = [intercept+coeff*i for i in X.ravel()]
# points = np.ravel([model_plot(i) for i in points])
# plt.plot(points,'g')
#
# #matplotlib scatter funcion w/ logistic regression
# plt.plot(X,'rx')
# plt.plot(Y,'bo')
# plt.xlabel("Credit Balance")
# plt.ylabel("Probability of Default")
# # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
# plt.legend(["Logistic Regression Model","X","Y"],
#            loc="lower right", fontsize='small')
# plt.show()

# ---------------------------------------------------
# # classification - random forest
# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# data = pd.read_csv(r"C:\Users\SKL\Documents\DLMLPYRTRAINING\Day3\Datasets\Classification.csv")
#
# #first we'll have to convert the strings "Single_Digit" and "Double_Digit" to numeric values
# data.loc[data["Class"] == "Single_Digit", "Class"] = 0
# data.loc[data["Class"] == "Double_Digit", "Class"] = 1
# X = data["Number"][:-2].values.reshape(-1, 1)
# Y = data["Class"][:-2].values.reshape(-1, 1)
#
# x_test = data["Number"][-2:].values.reshape(-1, 1)
# y_test = data["Class"][-2:].values.reshape(-1, 1)
#
# # build model
# model = RandomForestClassifier()
#
# # train model
# model.fit(X,Y.ravel())
#
# # score model
# score = model.score(X,Y.ravel())
# print("Score:\n", score)
#
# predict = model.predict(x_test)
# print("Prediction Feature:\n", x_test)
# print("Prediction Value:\n", predict)
# print("Actual Value:\n",y_test)

# -----------------------------------------------------------------
# classification - decision tree
#
# from sklearn.datasets import load_iris
# from sklearn.model_selection import cross_val_score
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(random_state=0)
# iris = load_iris()
# print(cross_val_score(clf, iris.data, iris.target, cv=10))
# -----------------------------------------------------------

# from sklearn.tree import DecisionTreeClassifier
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.model_selection import cross_val_score
#
# data = pd.read_csv(r"C:\Users\SKL\Documents\DLMLPYRTRAINING\Day3\Datasets\Classification.csv")
#
# #first we'll have to convert the strings "Single_Digit" and "Double_Digit" to numeric values
# data.loc[data["Class"] == "Single_Digit", "Class"] = 0
# data.loc[data["Class"] == "Double_Digit", "Class"] = 1
# X = data["Number"][:-2].values.reshape(-1, 1)
# Y = data["Class"][:-2].values.reshape(-1, 1)
#
# x_test = data["Number"][-2:].values.reshape(-1, 1)
# y_test = data["Class"][-2:].values.reshape(-1, 1)
#
# # build model
# model = DecisionTreeClassifier()
# # train model
# model.fit(X,Y)
#
# # score model
# score = model.score(X,Y)
# print("Score:\n", score)
#
# predict = model.predict(x_test)
# print("Prediction Feature:\n", x_test)
# print("Prediction Value:\n", predict)
# print("Actual Value:\n",y_test)

# ------------------------------------------------------------------
# # Clustering - Kmeans
# # import
# from sklearn.cluster import KMeans
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# X = np.array([[1, 2], [1, 4], [1, 0],
#               [4, 2], [4, 4], [4, 0]])
#
# # X = np.array([1,2,3,4,5,6,20,21,22,23,24,50,51,52,53,54,55])
# # X = X.reshape(-1,1)
#
# kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
#
# print(kmeans.labels_)
#
# print(kmeans.predict([[0, 0], [4, 4]]))
# # print(kmeans.predict([[9], [30], [60]]))
#
# print(kmeans.cluster_centers_)
#
# plt.plot(X[:,0],X[:,1],'bx')
# plt.plot(0,0,'rx')
# plt.plot(4,4,'rx')
# plt.plot(1,2,'gx')
# plt.plot(4,2,'gx')
# plt.show()

# ----------------------------------------------------------------------
# high end datasets
# http://archive.ics.uci.edu/ml/datasets.html?task=reg
#
# https://people.sc.fsu.edu/~jburkardt/datasets/regression/regression.html
#
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/frame.html
#
# http://users.stat.ufl.edu/~winner/datasets.html
#
# import pandas as pd
#
# data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data",header=None)
# print(data)
#
# data.loc[data[1]=='?',1]=5
# print(data)
#
# import pandas as pd
# # from sklearn.datasets import load_iris
# #
# # iris = load_iris()
# #
# df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', header=None)
#
# df.columns = ['ColA','ColB','ColC','ColD','ColE','ColF','ColG']
#
# df.rename(columns={'ColA':'C1','ColB':'C2','ColC':'C3','ColD':'C4','ColE':'C5','ColF':'C6','ColG':'C7'}, inplace=True)
#
# df = df.rename(columns={'ColA':'C11','ColB':'C22','ColC':'C33','ColD':'C44','ColE':'C55','ColF':'C66','ColG':'C77'})
