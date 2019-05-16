import pandas
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np

data_set = pandas.read_csv('C:\\Users\\dasmohap\\PycharmProjects\\All_ML_Python_Models\\K_Fold_CV\\dataset\\housing.csv')
print(data_set.head())
X = data_set.iloc[:, [0, 12]]
y = data_set.iloc[:, 13]

# Using mean Max scaller bring the dataset to normalized scale
scaler_val = MinMaxScaler(feature_range=(0, 1))

X = scaler_val.fit_transform(X)

scores = []
best_svr = SVR(kernel='rbf')
cv = KFold(n_splits=10, random_state=42, shuffle=False)
K=1
for train_index, test_index in cv.split(X):
    print("-"*200)
    print("K=",K,"\nTrain Index: ", train_index, "\n")
    print("K=",K,"\nTest Index: ", test_index)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_svr.fit(X_train, y_train)
    scores.append(best_svr.score(X_test, y_test))
    K=K+1

print("-"*200)
print("All Score: ",scores)
print("Mean Error: ",np.mean(scores))
exit(0)
#cross_val_score(best_svr, X, y, cv=10)
