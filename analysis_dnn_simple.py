import os
import heapq
import pickle
import innvestigate
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from load_data import LoadData
from gen_matrix import GenMatrix
from parse_file import ParseFile
from keras_rand_nn import RandNN, RunRandNN
from matplotlib.backends.backend_pdf import PdfPages


class Analyse():
    def __init__(self, dir_opt):
        self.dir_opt = dir_opt

    def plot_train_real_pred(self, path, epoch_time):
        # ALL POINTS PREDICTION SCATTERPLOT
        dir_opt = self.dir_opt
        pred_dl_input_df = pd.read_csv(path + '/PredTrainingInput.txt', delimiter = ',')
        # CALCULATE THE MSE FOR TRAIN
        train_auc_list = list(pred_dl_input_df['AUC'])
        train_auc = np.array(train_auc_list)
        train_pred_list = list(pred_dl_input_df['Pred Score'])
        train_pred = np.array(train_pred_list)
        train_mse = mean_squared_error(train_auc, train_pred)
        print('Training MSE: ' + str(train_mse))
        print(pred_dl_input_df.corr(method = 'pearson'))
        title = 'Scatter Plot After ' + epoch_time + ' Iterations In Training Dataset'
        ax = pred_dl_input_df.plot(x = 'AUC', y = 'Pred Score',
                    style = 'o', legend = False, title = title)
        ax.set_xlabel('AUC')
        ax.set_ylabel('Pred AUC')
        # SAVE TRAINING PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_train'
        path = '.' + dir_opt + '/plot/%s' % (file_name) + '.png'
        unit = 1
        if os.path.exists('.' + dir_opt + '/plot') == False:
            os.mkdir('.' + dir_opt + '/plot')
        while os.path.exists(path):
            path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(path, dpi = 300)
        
    def plot_test_real_pred(self, path, epoch_time):
        # ALL POINTS PREDICTION SCATTERPLOT
        dir_opt = self.dir_opt
        pred_dl_input_df = pd.read_csv(path + '/PredTestInput.txt', delimiter = ',')
        # CALCULATE THE MSE FOR TEST
        test_auc_list = list(pred_dl_input_df['AUC'])
        test_auc = np.array(test_auc_list)
        test_pred_list = list(pred_dl_input_df['Pred Score'])
        test_pred = np.array(test_pred_list)
        test_mse = mean_squared_error(test_auc, test_pred)
        print('Test MSE: ' + str(test_mse))
        print(pred_dl_input_df.corr(method = 'pearson'))
        title = 'Scatter Plot After ' + epoch_time + ' Iterations In Validation Dataset'
        ax = pred_dl_input_df.plot(x = 'AUC', y = 'Pred Score',
                    style = 'o', legend = False, title = title)
        ax.set_xlabel('AUC')
        ax.set_ylabel('Pred AUC')
        # SAVE TEST PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_validation'
        path = '.' + dir_opt + '/plot/%s' % (file_name) + '.png'
        unit = 1
        while os.path.exists(path):
            path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(path, dpi = 300)
    
    


if __name__ == "__main__":
    dir_opt = '/datainfo'
    epoch_time = '100'
    path = '.' + dir_opt + '/result/epoch_100_DNN_3'
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm2'
    pathway_filename = 'Selected_Kegg_Pathways2'
    post_data_path = '.' + dir_opt + '/post_data' # important may change xTe, xTr


    # ANALYSE DRUG EFFECT
    print('ANALYSING DRUG EFFECT...')
    Analyse(dir_opt).plot_train_real_pred(path, epoch_time)
    Analyse(dir_opt).plot_test_real_pred(path, epoch_time)
