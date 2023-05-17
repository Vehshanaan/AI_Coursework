import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
#from sklearn.datasets import load_boston

# 加载数据
save_path_X = "提供的数据与模板\X.npy"
save_path_Y = "提供的数据与模板\Y.npy"
X = np.load(save_path_X)
Y = np.load(save_path_Y)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.3],
    'kernel': ['linear', 'rbf']
}

# 创建SVR模型
svr = SVR()

# 创建GridSearchCV对象并进行参数搜索
grid_search = GridSearchCV(svr, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出最佳参数组合和对应的性能评分
print("Best Parameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

'''
Best Parameters:  {'C': 10, 'epsilon': 0.1, 'kernel': 'rbf'}
Best Score:  0.9431593340285088
'''