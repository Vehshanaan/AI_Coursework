'''
Author: “Vehshanaan” 1959180242@qq.com
Date: 2023-05-18 17:19:24
LastEditors: “Vehshanaan” 1959180242@qq.com
LastEditTime: 2023-05-18 18:15:16
FilePath: \AI_Coursework\代码\SVR.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR

# 加载数据
save_path_X = "提供的数据与模板\X.npy"
save_path_Y = "提供的数据与模板\Y.npy"
X = np.load(save_path_X)
Y = np.load(save_path_Y)

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割训练和验证集
Xtr, Xtest, Ytr, Ytest = train_test_split(X, Y, test_size=0.2)

svr = SVR()

param_grid = {
   "kernel": ["rbf"],
   "gamma": ["scale","auto"],
   "C": np.arange(7,10,0.5),
   "epsilon": np.arange(0.5,1,0.05)
}

# 创建交叉验证对象
cv = KFold(n_splits=2,shuffle=True,random_state=42)

# 创建GridSearchCV对象
grid_search = GridSearchCV(svr, param_grid,cv=cv,scoring="r2",verbose=1)#"neg_mean_squared_error")
grid_search.fit(Xtr,Ytr)


best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(best_params)
print(best_score)