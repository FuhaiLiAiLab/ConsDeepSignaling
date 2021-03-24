import os
import heapq
import pickle
import scipy
import innvestigate
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error

from load_data_flabel import LoadData
from gen_matrix import GenMatrix
from parse_file_flabel import ParseFile
from keras_rand_nn_flabel import RandNN, RunRandNN
from matplotlib.backends.backend_pdf import PdfPages



class Plot():
    def __init__(self, dir_opt):
        self.dir_opt = dir_opt
        
    def kde_corr_train(self, path, epoch_time):
        # ALL POINTS PREDICTION SCATTERPLOT
        dir_opt = self.dir_opt
        pred_dl_input_df = pd.read_csv(path + '/PredTrainingInput_flabel.txt', delimiter = ',')
        print(pred_dl_input_df.corr(method = 'pearson'))
        # CALCULATE THE MSE FOR TRAIN
        train_auc_list = list(pred_dl_input_df['AUC'])
        train_auc = np.array(train_auc_list)
        train_pred_list = list(pred_dl_input_df['Pred Score'])
        train_pred = np.array(train_pred_list)
        train_false_list = list(pred_dl_input_df['False AUC'])
        train_false = np.array(train_false_list)

        # train_auc_series = pd.Series(train_auc)
        # ax = train_auc_series.plot.kde()
        # plt.show()

        # train_pred_series = pd.Series(train_pred)
        # ax = train_pred_series.plot.kde()
        # plt.show()

        # train_series = pd.DataFrame({
        #     'AUC': train_auc,
        #     'Pred': train_pred
        # })
        # ax = train_series.plot.kde()
        # plt.xlim([0, 1])
        # plt.show()

        # s = pd.Series(train_auc_list)
        # ax = s.plot(kind = 'kde', style = 'k--')
        # plt.show()

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_x = [left, bottom + height + spacing, width, 0.2]
        rect_y = [left + width + spacing, bottom, 0.2, height]

        # start with a rectangular Figure
        plt.figure(figsize=(8, 8))

        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.tick_params(direction='in', top=True, right=True)
        ax_x = plt.axes(rect_x)
        ax_x.tick_params(direction='in', labelbottom=False)
        ax_y = plt.axes(rect_y)
        ax_y.tick_params(direction='in', labelleft=False)

        ax_scatter.scatter(train_auc, train_pred)
        ax_scatter.set_xlabel('AUC')
        ax_scatter.set_ylabel('Pred AUC')

        train_auc_series = pd.Series(train_auc)
        ax_x = train_auc_series.plot.kde()
        train_pred_series = pd.Series(train_pred)
        ax_y = train_pred_series.plot(kind='kde')

        plt.show()


        # # PLOT TRAIN RESULT
        # title = 'Scatter Plot After ' + epoch_time + ' Iterations In Training Dataset'
        # ax = pred_dl_input_df.plot(x = 'AUC', y = 'Pred Score',
        #             style = 'o', legend = False, title = title)
        # ax.set_xlabel('AUC')
        # ax.set_ylabel('Pred AUC')
        # # SAVE TRAINING PLOT FIGURE
        # file_name = 'epoch_' + epoch_time + '_train'
        # path = '.' + dir_opt + '/plot/%s' % (file_name) + '.png'
        # unit = 1
        # if os.path.exists('.' + dir_opt + '/plot') == False:
        #     os.mkdir('.' + dir_opt + '/plot')
        # while os.path.exists(path):
        #     path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.png'
        #     unit += 1
        # plt.savefig(path, dpi = 300)
        
    def kde_corr_test(self, path, epoch_time):
        # ALL POINTS PREDICTION SCATTERPLOT
        dir_opt = self.dir_opt
        pred_dl_input_df = pd.read_csv(path + '/PredTestInput_flabel.txt', delimiter = ',')
        print(pred_dl_input_df.corr(method = 'pearson'))
        # CALCULATE THE MSE FOR TEST
        test_auc_list = list(pred_dl_input_df['AUC'])
        test_auc = np.array(test_auc_list)
        test_pred_list = list(pred_dl_input_df['Pred Score'])
        test_pred = np.array(test_pred_list)
        test_false_list = list(pred_dl_input_df['False AUC'])
        test_false = np.array(test_false_list)
        # PLOT TEST RESULT 
        title = 'Scatter Plot After ' + epoch_time + ' Iterations In Test Dataset'
        ax = pred_dl_input_df.plot(x = 'AUC', y = 'Pred Score',
                    style = 'o', legend = False, title = title)
        ax.set_xlabel('AUC')
        ax.set_ylabel('Pred AUC')
        # SAVE TEST PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_test'
        path = '.' + dir_opt + '/plot/%s' % (file_name) + '.png'
        unit = 1
        while os.path.exists(path):
            path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(path, dpi = 300)

    

    



if __name__ == "__main__":
    dir_opt = '/datainfo'
    epoch_time = '100'
    path = '.' + dir_opt + '/result/5-fold-flabel01/epoch_99'
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm2'
    pathway_filename = 'Selected_Kegg_Pathways2'
    post_data_path = '.' + dir_opt + '/post_data' # important may change xTe, xTr


    # # ANALYSE DRUG EFFECT
    print('ANALYSING DRUG EFFECT...')
    Plot(dir_opt).kde_corr_train(path, epoch_time)


    # Plot(dir_opt).kde_corr_test(path, epoch_time)
