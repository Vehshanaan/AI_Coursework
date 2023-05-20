from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor


# Load data
save_path_X = "提供的数据与模板\X.npy"
save_path_Y = "提供的数据与模板\Y.npy"
X = np.load(save_path_X)
Y = np.load(save_path_Y)

# Split data
Xtr, Xtest, Ytr, Ytest = train_test_split(X, Y, test_size=0.2)

# Dummy
dummy_regr = DummyRegressor()
dummy_regr.fit(Xtr, Ytr)

Ytest_pred = dummy_regr.predict(Xtest)

mse_dummy = mean_squared_error(Ytest, Ytest_pred)
mae_dummy = mean_absolute_error(Ytest, Ytest_pred)
r2_dummy = r2_score(Ytest, Ytest_pred)

print("Dummy:")
print("mse: "+str(mse_dummy))
print("mae: "+str(mae_dummy))
print("r2: "+str(r2_dummy))
print("----------------------\n")

# KNR
scaler = StandardScaler()
scaler.fit(X)
Xtest_scaled = scaler.transform(Xtest)
Xtr_scaled = scaler.transform(Xtr)

knr = KNeighborsRegressor(n_neighbors=7, p=1, weights="distance")

knr.fit(Xtr_scaled, Ytr)

Ytest_pred = knr.predict(Xtest_scaled)

mse_KNR = mean_squared_error(Ytest, Ytest_pred)
mae_KNR = mean_absolute_error(Ytest, Ytest_pred)
r2_KNR = r2_score(Ytest, Ytest_pred)

print("KNR:")
print("mse: "+str(mse_KNR))
print("mae: "+str(mae_KNR))
print("r2: "+str(r2_KNR))
print("----------------------\n")

# DTR

dtr = DecisionTreeRegressor(
    random_state=42, max_depth=14, min_samples_leaf=13, splitter="best")

dtr.fit(Xtr, Ytr)

Ytest_pred = dtr.predict(Xtest)

mse_DTR = mean_squared_error(Ytest, Ytest_pred)
mae_DTR = mean_absolute_error(Ytest, Ytest_pred)
r2_DTR = r2_score(Ytest, Ytest_pred)

print("DTR:")
print("mse: "+str(mse_DTR))
print("mae: "+str(mae_DTR))
print("r2: "+str(r2_DTR))
print("----------------------\n")
