import os
import numpy as np
import pandas as pd
from numpy import savetxt
from sklearn.metrics import mean_squared_error

class Output():
    def __init__(self, dir_opt):
        self.dir_opt = dir_opt

    def combine_pas_net_output(self, folder_name, model_name):
        print('\n------- PASnet PREDICTION RESULTS -------')
        test_pred_path = '.' + dir_opt + '/%s/%s/PredTestInput.txt' % (folder_name, model_name)
        test_pred_df = pd.read_table(test_pred_path, delimiter = ',')
        merged_pred_df = test_pred_df
        for unit in range(1, 5):
            next_test_pred_path = '.' + dir_opt + '/%s/%s_%d/PredTestInput.txt' % (folder_name, model_name, unit)
            next_test_pred_df = pd.read_table(next_test_pred_path, delimiter = ',')
            merged_pred_df = pd.concat([merged_pred_df, next_test_pred_df])
        merged_pred_df = merged_pred_df.reset_index(drop = True)
        print(merged_pred_df)
        merged_pred_df.to_csv('./datainfo/result/5-fold/PASnet-prediction.csv', index = False, header = True)
        # CALCULATE THE MSE AND PEARSON CORRELATION FOR TEST DATA SETS
        test_auc_list = list(merged_pred_df['AUC'])
        test_auc = np.array(test_auc_list)
        test_pred_list = list(merged_pred_df['Pred Score'])
        test_pred = np.array(test_pred_list)
        test_mse = mean_squared_error(test_auc, test_pred)
        print('Test MSE: ' + str(test_mse))
        print(merged_pred_df.corr(method = 'pearson'))
        

if __name__ == "__main__":
    dir_opt = '/datainfo'
    folder_name = 'result/5-fold'

    # OUTPUT PATH-DNN COMBINED PREDICTION
    model_name = 'epoch_99'
    Output(dir_opt).combine_pas_net_output(folder_name, model_name)