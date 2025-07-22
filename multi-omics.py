from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from time import time
from glmnet import ElasticNet as GLMNet

print("Reading data...")
script_start_time = int(time())
start_time = time()

data = pd.read_csv("./X.csv", sep=";", index_col="index")
X = data.transpose().values
index = data.index
print(f"X read: %.2f seconds" % (time() - start_time))
start_time = time()

y = pd.read_csv("./Y.csv", sep=";")["Age (years)"].values
print(f"y read: %.2f seconds" % (time() - start_time))
start_time = time()

counter = 0
cv = LeaveOneOut()
y_true, y_pred = list(), list()
for train_ix, test_ix in cv.split(X):
    counter += 1
    print(f"Fold #{counter}...")

    X_train, X_test = X[train_ix, :], X[test_ix, :]
    y_train, y_test = y[train_ix], y[test_ix]

    model = GLMNet(alpha=0.01, random_state=30)
    model.fit(X_train, y_train.ravel())
    pred = model.predict(X_test)
    y_true += list(y_test)
    y_pred += list(pred)
    print(f"  MAE: {mean_absolute_error(y_test, pred)}")

    coef = []
    for i in range(len(model.coef_)):
        if model.coef_[i] != 0:
            coef.append((index[i], model.coef_[i]))
    coef_rows = list(map(lambda x: f"{x[0]};{x[1]}", coef))
    f = open(f"./multi-result-coef-fold_{counter}-{script_start_time}.csv", "w")
    f.write(f"#intercept = {model.intercept_}\nid;coef\n" + "\n".join(coef_rows))
    f.close()

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
pearson, p = pearsonr(y_true, y_pred)
f = open(f"./multi-result-stats-{script_start_time}.csv", "w")
f.write(f"mae;mse;pearson\n{mae};{mse};{pearson}\n")
f.close()

y_combined = list(zip(y_true, y_pred))
f = open(f"./multi-result-pred-{script_start_time}.csv", "w")
f.write("y_true;y_pred\n")
for z in y_combined:
    f.write(f"{z[0]};{z[1]}\n")
f.close()

print("Done!")