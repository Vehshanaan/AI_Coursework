'''
Author: “Vehshanaan” 1959180242@qq.com
Date: 2023-05-18 12:17:16
LastEditors: “Vehshanaan” 1959180242@qq.com
LastEditTime: 2023-05-20 16:28:46
FilePath: \AI_Coursework\代码\KNN_iter.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
'''
Author: “Vehshanaan” 1959180242@qq.com
Date: 2023-05-17 15:26:31
LastEditors: “Vehshanaan” 1959180242@qq.com
LastEditTime: 2023-05-18 15:08:51
FilePath: \AI_Coursework\代码\KNN_iter.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold

import matplotlib.pyplot as plt


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

# 初始化knr和param
knr = KNeighborsRegressor()
param_grid = {
    'n_neighbors': range(1,20),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

# 创建交叉验证对象
cv = KFold(n_splits=10,shuffle=True,random_state=42)

# 创建GridSearchCV对象
grid_search = GridSearchCV(knr, param_grid,cv=cv,scoring="neg_mean_squared_error",verbose=1)#"neg_mean_squared_error")
grid_search.fit(Xtr,Ytr)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(best_params)
print(best_score)


# 获取每个参数组合的得分
scores = grid_search.cv_results_['mean_test_score']

# 获取每个参数组合的参数值
param_values = grid_search.cv_results_['params']

# 提取每个参数的取值范围
n_neighbors = [param['n_neighbors'] for param in param_values]
weights = [param['weights'] for param in param_values]
p = [param['p'] for param in param_values]



# 将性能随参数变化的曲线绘制成图表
fig = plt.figure(figsize=(12, 6))

# 绘制n_neighbors对性能的影响
plt.subplot(1, 3, 1)
plt.scatter(n_neighbors, scores)
plt.xlabel('n_neighbors')
plt.ylabel('Negative MSE')
plt.title('Performance vs n_neighbors')
plt.ylim(-16,-12.5)

# 绘制weights对性能的影响
plt.subplot(1, 3, 2)
plt.scatter(weights, scores)
plt.xlabel('weights')
plt.ylabel('Negative MSE')
plt.title('Performance vs weights')
plt.ylim(-16,-12.5)

# 绘制p对性能的影响
plt.subplot(1, 3, 3)
plt.scatter(p, scores)
plt.xlabel('p')
plt.ylabel('Negative MSE')
plt.title('Performance vs p')
plt.ylim(-16,-12.5)


plt.tight_layout()

plt.show()




'''


mse = mean_squared_error(Ytest, Ytest_pred)
mae = mean_absolute_error(Ytest, Ytest_pred)
r2 = r2_score(Ytest, Ytest_pred)


print("MSE = "+str(mse))
print("MAE = "+str(mae))
print("R2 = "+str(r2))
'''
