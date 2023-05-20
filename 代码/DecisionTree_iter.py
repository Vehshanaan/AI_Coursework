'''
Author: “Vehshanaan” 1959180242@qq.com
Date: 2023-05-18 16:22:27
LastEditors: “Vehshanaan” 1959180242@qq.com
LastEditTime: 2023-05-20 15:09:57
FilePath: \AI_Coursework\代码\DecisionTree_iter.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
#https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# 加载数据
save_path_X = "提供的数据与模板\X.npy"
save_path_Y = "提供的数据与模板\Y.npy"
X = np.load(save_path_X)
Y = np.load(save_path_Y)

# 标准化数据
#scaler = StandardScaler()
#X = scaler.fit_transform(X)

# 分割训练和验证集
Xtr, Xtest, Ytr, Ytest = train_test_split(X, Y, test_size=0.2)


# 创建交叉验证对象
cv = KFold(n_splits=10,shuffle=True,random_state=42)

dtr = DecisionTreeRegressor()

param_grid = {
    'max_depth' : range(5,20),
    "min_samples_leaf": range(1,30),
    "splitter": ['best','random'],
    "random_state": [42]
}

grid_search = GridSearchCV(dtr, param_grid, cv = cv, scoring = "neg_mean_squared_error", verbose=1)
grid_search.fit(Xtr,Ytr)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(best_params)
print(best_score)

# 获取每个参数组合的得分
scores = grid_search.cv_results_['mean_test_score']

# 获取每个参数组合的参数值
param_values = grid_search.cv_results_['params']

max_depths = [param['max_depth'] for param in param_values]
min_samples_leafs = [param['min_samples_leaf'] for param in param_values]
splitters = [param['splitter'] for param in param_values]


# 将性能随参数变化的曲线绘制成图表
fig = plt.figure(figsize=(12, 6))

# 绘制n_neighbors对性能的影响
plt.subplot(1, 3, 1)
plt.scatter(max_depths, scores)
plt.xlabel('max_depths')
plt.ylabel('Score')
plt.title('Performance vs max_depths')
plt.ylim(-18,-16)

# 绘制weights对性能的影响
plt.subplot(1, 3, 2)
plt.scatter(min_samples_leafs, scores)
plt.xlabel('min_samples_leafs')
plt.ylabel('Score')
plt.title('Performance vs min_samples_leafs')
plt.ylim(-18,-16)

# 绘制p对性能的影响
plt.subplot(1, 3, 3)
plt.scatter(splitters, scores)
plt.xlabel('splitters')
plt.ylabel('Score')
plt.title('Performance vs splitters')
plt.ylim(-18,-16)


plt.tight_layout()
plt.show()


