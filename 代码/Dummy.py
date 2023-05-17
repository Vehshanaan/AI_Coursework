import numpy as np


# 加载数据
save_path_X = "提供的数据与模板\X.npy"
save_path_Y = "提供的数据与模板\Y.npy"
X = np.load(save_path_X)
Y = np.load(save_path_Y)

# 分割训练和验证集

from sklearn.model_selection import train_test_split
Xtr, Xtest, Ytr, Ytest = train_test_split(X,Y,test_size=0.2)

# 导入dummy回归器

from sklearn.dummy import DummyRegressor

dummy_regr = DummyRegressor()
dummy_regr.fit(Xtr,Ytr)

Ytest_pred = dummy_regr.predict(Xtest)


# 常用的回归性能指标：MAE和MSE都是越小越好，然后R2是越接近1越好
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


mse = mean_squared_error(Ytest, Ytest_pred)
mae = mean_absolute_error(Ytest, Ytest_pred)
r2 = r2_score(Ytest, Ytest_pred)


print("MSE = "+str(mse))
print("MAE = "+str(mae))
print("R2 = "+str(r2))