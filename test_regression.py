import os
import numpy as np
import pandas as pd
import _pickle as cPickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def load_data():
    xTr = np.load('./datainfo/post_data/xTr.npy')
    yTr = np.load('./datainfo/post_data/yTr.npy')
    xTe = np.load('./datainfo/post_data/xTe.npy')
    yTe = np.load('./datainfo/post_data/yTe.npy')
    print(xTr.shape)
    print(xTe.shape)
    return xTr, yTr, xTe, yTe

def ranfore_regression(xTr, yTr, xTe, max_depth):
    if max_depth == "none":
        # regr = RandomForestRegressor(random_state = 0)
        regr = RandomForestRegressor()
    else:
        # regr = RandomForestRegressor(max_depth = max_depth, random_state = 0)
        regr = RandomForestRegressor(max_depth = max_depth)
    regr.fit(xTr, yTr)
    y_train_pred_list = list(regr.predict(xTr))
    print(y_train_pred_list)
    y_test_pred_list = list(regr.predict(xTe))
    print(y_test_pred_list)
    # PRESERVE THE MODEL TO FILE
    # CLEAN RESULT PREVIOUS EPOCH_I_PRED FILES
    folder_name = 'ranfore_max_depth_' + str(max_depth)
    path = './datainfo/regression_result/%s' % (folder_name)
    unit = 1
    if os.path.exists('./datainfo/regression_result') == False:
        os.mkdir('./datainfo/regression_result')
    while os.path.exists(path):
        path = './datainfo/regression_result/%s_%d' % (folder_name, unit)
        unit += 1
    os.mkdir(path)
    with open(path + '/ranfore_model', 'wb') as f:
        cPickle.dump(regr, f)
    return y_train_pred_list, y_test_pred_list

def ranfore_pred(path, xTr, xTe):
    with open(path + '/ranfore_model', 'rb') as f:
        regr = cPickle.load(f)
    y_train_pred_list = list(regr.predict(xTr))
    print(y_train_pred_list)
    y_test_pred_list = list(regr.predict(xTe))
    print(y_test_pred_list)
    return y_train_pred_list, y_test_pred_list


def ridge_regression(xTr, yTr, xTe):
    ridge = Ridge(alpha = 1.0, max_iter = 50).fit(xTr, yTr)
    xTr_length, _ = xTr.shape
    xTe_length, _ = xTe.shape
    y_train_pred_list = np.reshape(ridge.predict(xTr), xTr_length)
    print(y_train_pred_list)
    y_test_pred_list = np.reshape(ridge.predict(xTe), xTe_length)
    print(y_test_pred_list)
    return y_train_pred_list, y_test_pred_list

def lasso_regression(xTr, yTr, xTe):
    lasso = linear_model.Lasso(alpha = 0.0001, max_iter = 50).fit(xTr, yTr)
    xTr_length, _ = xTr.shape
    xTe_length, _ = xTe.shape
    y_train_pred_list = np.reshape(lasso.predict(xTr), xTr_length)
    print(y_train_pred_list)
    y_test_pred_list = np.reshape(lasso.predict(xTe), xTe_length)
    print(y_test_pred_list)
    return y_train_pred_list, y_test_pred_list


def linear_regression(xTr, yTr, xTe):
    reg = LinearRegression().fit(xTr, yTr)
    xTr_length, _ = xTr.shape
    xTe_length, _ = xTe.shape
    y_train_pred_list = np.reshape(reg.predict(xTr), xTr_length)
    print(y_train_pred_list)
    y_test_pred_list = np.reshape(reg.predict(xTe), xTe_length)
    print(y_test_pred_list)
    return y_train_pred_list, y_test_pred_list

def train_result(y_train_pred_list):
    # CALCULATE THE MSE AND PEARSON CORRELATION FOR TEST
    final_train_input_df = pd.read_csv('./datainfo/filtered_data/TrainingInput.txt', delimiter = ',')
    final_row, final_col = final_train_input_df.shape
    final_train_input_df.insert(final_col, 'Pred Score', y_train_pred_list, True)

    train_auc_list = list(final_train_input_df['AUC'])
    train_auc = np.array(train_auc_list)
    train_pred_list = list(final_train_input_df['Pred Score'])
    train_pred = np.array(train_pred_list)
    train_mse = mean_squared_error(train_auc, train_pred)
    print('\nTraining MSE: ' + str(train_mse))
    print('Training Pearson: ' + str(final_train_input_df.corr(method = 'pearson')))

def test_result(y_test_pred_list):
    # CALCULATE THE MSE AND PEARSON CORRELATION FOR TEST
    final_test_input_df = pd.read_csv('./datainfo/filtered_data/TestInput.txt', delimiter = ',')
    final_row, final_col = final_test_input_df.shape
    final_test_input_df.insert(final_col, 'Pred Score', y_test_pred_list, True)

    test_auc_list = list(final_test_input_df['AUC'])
    test_auc = np.array(test_auc_list)
    test_pred_list = list(final_test_input_df['Pred Score'])
    test_pred = np.array(test_pred_list)
    test_mse = mean_squared_error(test_auc, test_pred)
    print('\nTest MSE: ' + str(test_mse))
    print('Test Pearson: ' + str(final_test_input_df.corr(method = 'pearson')))


if __name__ == "__main__":

    if os.path.exists('./datainfo/regression_result') == False:
            os.mkdir('./datainfo/regression_result')

    xTr, yTr, xTe, yTe = load_data()

    # # RIDGE REGRESSION
    # print("\n------------ RIDGE REGRESSION ------------")
    # y_train_pred_list, y_test_pred_list = ridge_regression(xTr, yTr, xTe)
    # train_result(y_train_pred_list)
    # test_result(y_test_pred_list)

    # # LASSO REGRESSION
    # print("\n------------ LASSO REGRESSION ------------")
    # y_train_pred_list, y_test_pred_list = lasso_regression(xTr, yTr, xTe)
    # train_result(y_train_pred_list)
    # test_result(y_test_pred_list)

    # # LINEAR REGRESSION
    # print("\n------------ LINEAR REGRESSION ------------")
    # y_train_pred_list, y_test_pred_list = linear_regression(xTr, yTr, xTe)
    # train_result(y_train_pred_list)
    # test_result(y_test_pred_list)

    # RANDOM FOREST REGRESSION
    print("\n------------ RANDOM FOREST REGRESSION ------------")
    max_depth = "none"
    # max_depth = 2
    max_features = "auto" # {"auto", "sqrt”, "log2"}
    y_train_pred_list, y_test_pred_list = ranfore_regression(xTr, yTr, xTe, max_depth)
    # path = './datainfo/regression_result/ranfore_max_depth_none'
    # y_train_pred_list, y_test_pred_list = ranfore_pred(path, xTr, xTe)
    train_result(y_train_pred_list)
    test_result(y_test_pred_list)