import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import joblib


def rmse(pred, label):
    tot = 0
    for i in range(len(pred)):
        rmse = 0
        for j in range(len(pred[i])):
            rmse += (pred[i][j] - label[i][j])**2
        rmse = np.sqrt(rmse / (len(pred[i])))
        tot += rmse

    return tot / (len(pred))


# 输入数据
# tot_path = r'D:\VScodeProjects\PythonProjects\ColorCorrection\Code\Dataset\result.xlsx'
tot_path = r'D:\VScodeProjects\PythonProjects\ColorCorrection\Code\Dataset\tot_result.xlsx'
# train_Path = r'D:\VScodeProjects\PythonProjects\ColorCorrection\Code\Dataset\train.xlsx'
# test_Path = r'D:\VScodeProjects\PythonProjects\ColorCorrection\Code\Dataset\test.xlsx'
# 输出表格
result_path = r'D:\VScodeProjects\PythonProjects\ColorCorrection\Code\output.xlsx'

df = pd.read_excel(tot_path)
# df = df.loc[df['f_distance'] != 0.0]
# df = df.loc[df['mobile'] != 'vivo']
data = np.array(df)
train_data, test_data = train_test_split(data, train_size=0.8, random_state=42, shuffle=True)

# 载入输入数据
# df = pd.read_excel(train_Path)
# train_data = np.array(df)
train_input = train_data[:, 3:9]
# train_input = np.hstack((train_data[:, 3:9], train_data[:, 10:12], train_data[:, 12:14]))
train_input = train_input.astype(np.float32)

# 载入输出数据
# train_label = np.array(pd.read_csv(train_Path))
train_label = train_data[:, 15:18]
train_label = train_label.astype(np.float32)

# cc = list(zip(train_input, train_label))
# random.shuffle(cc)
# train_input[:], train_label[:] = zip(*cc)

# 载入测试集
# test_data = np.array(pd.read_excel(test_Path))
test_input = test_data[:, 3:9]
# test_input = np.hstack((test_data[:, 3:9], test_data[:, 10:12], test_data[:, 12:14]))
test_input = test_input.astype(np.float32)

# test_label = pd.read_csv(test_Path)
test_label = test_data[:, 15:18]
test_label = test_label.astype(np.float32)

random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=1)  # 随机森林回归,并使用50个决策树
random_forest_regressor.fit(train_input, train_label)  # 拟合模型
# joblib.dump(random_forest_regressor, './models/forest/forest.m')
feature_importances = random_forest_regressor.feature_importances_
print("feat importance = " + str(feature_importances))

train_pred = random_forest_regressor.predict(train_input)
test_pred = random_forest_regressor.predict(test_input)

print('train myRMSE:{:.3f}'.format(rmse(train_pred, train_label)))
print('train RMSE:{:.3f}'.format(np.sqrt(mean_squared_error(train_label, train_pred))))
print('train MAE:{:.3f}'.format(mean_absolute_error(train_label, train_pred)))
print('train R2:{:.3f}'.format(r2_score(train_pred, train_label)))
print('test myRMSE:{:.3f}'.format(rmse(test_pred, test_label)))
print('test RMSE:{:.3f}'.format(np.sqrt(mean_squared_error(test_pred, test_label))))
print('test MAE:{:.3F}'.format(mean_absolute_error(test_label, test_pred)))
print('test R2:{:.3f}'.format(r2_score(test_pred, test_label)))
