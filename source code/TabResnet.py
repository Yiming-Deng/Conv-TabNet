import random
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

from pytorch_widedeep import Trainer
from pytorch_widedeep.models import TabResnet, WideDeep, TabMlp, TabTransformer
from pytorch_widedeep.preprocessing import TabPreprocessor
from pytorch_widedeep.initializers import KaimingNormal


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

    # # 输入数据
    # train_Path = r'D:\VScodeProjects\PythonProjects\ColorCorrection\Code\Dataset\train.xlsx'
    # test_Path = r'D:\VScodeProjects\PythonProjects\ColorCorrection\Code\Dataset\test.xlsx'
    # # 输出表格
    # result_path = r'D:\VScodeProjects\PythonProjects\ColorCorrection\Code\output.xlsx'

    # # 载入输入数据
    # train_data = np.array(pd.read_excel(train_Path))
    # train_input = train_data[:, 3:10]
    # train_input = train_input.astype(np.float32)

    # # 载入输出数据
    # # train_label = np.array(pd.read_csv(train_Path))
    # train_label = train_data[:, 11:14]
    # train_label = train_label.astype(np.float32)

    # # print(train_input, train_label.shape)
    # cc = list(zip(train_input, train_label))
    # random.shuffle(cc)
    # train_input[:], train_label[:] = zip(*cc)
    # # print(train_input, train_label.shape)
    # # input()

    # # 载入测试集
    # test_data = np.array(pd.read_excel(test_Path))
    # test_input = test_data[:, 3:10]
    # test_input = test_input.astype(np.float32)

    # # test_label = pd.read_csv(test_Path)
    # test_label = test_data[:, 11:14]
    # test_label = test_label.astype(np.float32)

    df = pd.read_excel(r'.\Dataset\tot_result.xlsx')
    df = df.loc[df['f_distance'] != 0.0]
    train_data, test_data = train_test_split(df, train_size=0.8, random_state=42, shuffle=True)
    continuous_cols = ["R", "G", "B", "R_P", "G_P", "R_P"]
    target = ["R_L", "G_L", "B_L"]
    target = train_data[target].values

    tab_preprocessor = TabPreprocessor(continuous_cols=continuous_cols)
    X_tab = tab_preprocessor.fit_transform(train_data)

    test_input = tab_preprocessor.fit_transform(test_data)
    # print(test_input)
    # input()
    # test_data = np.array(test_data)
    # test_input = np.hstack((test_data[:, 3:9], test_data[:, 10:12], test_data[:, 12:14]))
    # test_input = test_input.astype(np.float32)

    test_data = np.array(test_data)
    test_label = test_data[:, 15:18]
    test_label = test_label.astype(np.float32)

    tabMlp = TabMlp(continuous_cols=continuous_cols,
                    mlp_batchnorm=True,
                    mlp_hidden_dims=[100, 5000, 5000, 100],
                    column_idx=tab_preprocessor.column_idx)
    tabResnet = TabResnet(continuous_cols=continuous_cols,
                          mlp_batchnorm=True,
                          blocks_dims=[200, 100, 100],
                          mlp_hidden_dims=[100, 5000, 5000, 100],
                          column_idx=tab_preprocessor.column_idx)
    tabTransformer = TabTransformer(continuous_cols=continuous_cols,
                                    embed_continuous=True,
                                    mlp_batchnorm=True,
                                    mlp_hidden_dims=[1000, 10000, 10000, 1000],
                                    shared_embed=True,
                                    column_idx=tab_preprocessor.column_idx)
    model = WideDeep(deeptabular=tabResnet, pred_dim=3)

    deep_opt = torch.optim.Adam(model.deeptabular.parameters())
    deep_sch = torch.optim.lr_scheduler.StepLR(deep_opt, step_size=3)
    optimizers = {"deeptabular": deep_opt}
    schedulers = {"deeptabular": deep_sch}
    initializers = {"deeptabular": KaimingNormal}

    trainer = Trainer(model,
                      objective='regression',
                      initializers=initializers,
                      optimizers=optimizers,
                      lr_schedulers=schedulers)
    trainer.fit(X_tab=X_tab, target=target, n_epochs=20, batch_size=64)
    test_preds = trainer.predict(X_tab=test_input, batch_size=64)
    # f_imp = trainer.explain(X_tab=test_input)
    # print(f_imp)
    # input()
    # trainer.save(path='./models/tabtransformer/', model_filename='tabTransformer.pt', save_state_dict=True)
    # torch.save(model.state_dict(), "./models/tabtransformer/tabTransformer.pt")

    # 评价
    print('test myRMSE:{:.3f}'.format(rmse(test_preds, test_label)))
    print('test RMSE:{:.3f}'.format(np.sqrt(mean_squared_error(test_preds, test_label))))
    print('test MAE:{:.3F}'.format(mean_absolute_error(test_label, test_preds)))
    print('test R2:{:.3f}'.format(r2_score(test_preds, test_label)))
