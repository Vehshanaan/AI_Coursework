'''
Author: “Vehshanaan” 1959180242@qq.com
Date: 2023-05-17 12:23:17
LastEditors: “Vehshanaan” 1959180242@qq.com
LastEditTime: 2023-05-19 10:29:45
FilePath: \AI_Coursework\代码\DataReader.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import numpy as np

data_path = "提供的数据与模板\coursework_other(1).csv"
save_path_X = "提供的数据与模板\X.npy"
save_path_Y = "提供的数据与模板\Y.npy"


# 返回一个numpy数组，csv中的每一行是一个元素。第一行的标签也读进去了
data = np.genfromtxt(data_path, delimiter=',',skip_header=1)

# 删除第一行的标签
print(data)

# 分离前面的值（X）和最后的预测目标PE值（Y）

X = data[:,:-1]
Y = data[:,-1]

# 保存数据
X = np.array(X)
Y = np.array(Y)

print(np.shape(X))

'''
np.save(save_path_X, X)
np.save(save_path_Y, Y)
'''


