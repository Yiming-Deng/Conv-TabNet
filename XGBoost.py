import numpy as np
import os
from pandas.core.frame import DataFrame
import pandas as pd
from visdom import Visdom
import math
import random
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from openpyxl import load_workbook
from openpyxl.styles import PatternFill, colors

from pytorch_tabnet.tab_model import TabNetRegressor

from sklearn.metrics import r2_score


def rmse(pred, label):
    tot = 0
    for i in range(len(pred)):
        rmse = 0
        for j in range(len(pred[i])):
            rmse += (pred[i][j] - label[i][j])**2
        rmse = np.sqrt(rmse / (len(pred[i])))
        tot += rmse

    return tot / (len(pred))


if __name__ == '__main__':
    batch_size = 64  # 64
    # 输入数据
    # tot_path = r'D:\VScodeProjects\PythonProjects\ColorCorrection\Code\Dataset\result.xlsx'
    tot_path = r'D:\VScodeProjects\PythonProjects\ColorCorrection\Code\Dataset\tot_result.xlsx'
    # train_Path = r'D:\VScodeProjects\PythonProjects\ColorCorrection\Code\Dataset\train.xlsx'
    # test_Path = r'D:\VScodeProjects\PythonProjects\ColorCorrection\Code\Dataset\test.xlsx'
    # 输出表格
    result_path = r'D:\VScodeProjects\PythonProjects\ColorCorrection\Code\output.xlsx'

    df = pd.read_excel(tot_path)
    df = df.loc[df['f_distance'] != 0.0]
    df = df.loc[df['mobile'] != 'vivo']
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
    # test_input = np.hstack((test_data[:, 3:9], test_data[:, 10:11], test_data[:, 12:14]))
    test_input = test_input.astype(np.float32)

    # test_label = pd.read_csv(test_Path)
    test_label = test_data[:, 15:18]
    test_label = test_label.astype(np.float32)

    # xgboost模型初始化设置
    dtrain = xgb.DMatrix(train_input, label=train_label)
    dtest = xgb.DMatrix(test_input)
    watchlist = [(dtrain, 'train')]

    # booster:
    params = {
        'booster': 'gbtree',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 5,  #5
        'lambda': 10,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'min_child_weight': 2,
        'eta': 0.025,  #0.025
        'seed': 0,
        'nthread': 8,
        'gamma': 0.15,  #0.15
        'learning_rate': 0.01
    }

    # 建模与预测：50棵树
    bst = xgb.Booster(model_file='XGB_model.xgb')  # 加载模型
    bst = xgb.train(params, dtrain, num_boost_round=20000, evals=watchlist)
    # bst.save_model('./models/xgb/XGB_model.xgb')
    train_pred = bst.predict(dtrain)
    y_pred = bst.predict(dtest)

    # 设置阈值、评价指标
    # y_pred = (ypred >= 0.5) * 1
    print('train myRMSE:{:.3f}'.format(rmse(train_pred, train_label)))
    print('train RMSE:{:.3f}'.format(np.sqrt(mean_squared_error(train_label, train_pred))))
    print('train MAE:{:.3f}'.format(mean_absolute_error(train_label, train_pred)))
    print('train R2:{:.3f}'.format(r2_score(train_pred, train_label)))
    print('test myRMSE:{:.3f}'.format(rmse(y_pred, test_label)))
    print('test RMSE:{:.3f}'.format(np.sqrt(mean_squared_error(y_pred, test_label))))
    print('test MAE:{:.3F}'.format(mean_absolute_error(test_label, y_pred)))
    print('test R2:{:.3f}'.format(r2_score(y_pred, test_label)))

    ypred = bst.predict(dtest)
    print("测试集每个样本的得分\n", ypred)
    ypred_leaf = bst.predict(dtest, pred_leaf=True)
    print("测试集每棵树所属的节点数\n", ypred_leaf)
    ypred_contribs = bst.predict(dtest, pred_contribs=True)
    print("特征的重要性\n", ypred_contribs)

    #设置字体为楷体
    # matplotlib.rcParams['font.sans-serif'] = ['KaiTi']

    # xgb.plot_importance(bst, height=0.8, title='影响校正效果的重要特征', ylabel='特征')
    # plt.rc('font', family='Arial Unicode MS', size=14)
    # plt.show()