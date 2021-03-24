import os
import heapq
import pickle
import innvestigate
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from matplotlib import gridspec
from sklearn.metrics import mean_squared_error

from load_data_flabel import LoadData
from gen_matrix import GenMatrix
from parse_file_flabel import ParseFile
from keras_rand_nn_flabel import RandNN, RunRandNN
from matplotlib.backends.backend_pdf import PdfPages



class Analyse():
    def __init__(self, dir_opt):
        self.dir_opt = dir_opt
        
    def plot_train_real_pred(self, path, epoch_time):
        # ALL POINTS PREDICTION SCATTERPLOT
        dir_opt = self.dir_opt
        pred_dl_input_df = pd.read_csv(path + '/PredTrainingInput_flabel.txt', delimiter = ',')
        print(pred_dl_input_df.corr(method = 'pearson'))
        # CALCULATE THE MSE FOR TRAIN
        train_auc_list = list(pred_dl_input_df['AUC'])
        train_auc = np.array(train_auc_list)
        train_pred_list = list(pred_dl_input_df['Pred Score'])
        train_pred = np.array(train_pred_list)
        train_mse = mean_squared_error(train_auc, train_pred)
        print('Training MSE: ' + str(train_mse))
        # PLOT TRAIN RESULT
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
        pred_dl_input_df = pd.read_csv(path + '/PredTestInput_flabel.txt', delimiter = ',')
        print(pred_dl_input_df.corr(method = 'pearson'))
        # CALCULATE THE MSE FOR TEST
        test_auc_list = list(pred_dl_input_df['AUC'])
        test_auc = np.array(test_auc_list)
        test_pred_list = list(pred_dl_input_df['Pred Score'])
        test_pred = np.array(test_pred_list)
        test_mse = mean_squared_error(test_auc, test_pred)
        print('Test MSE: ' + str(test_mse))
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

    def train_pvalue(self, path):
        # ALL POINTS PREDICTION SCATTERPLOT
        dir_opt = self.dir_opt
        pred_dl_input_df = pd.read_csv(path + '/PredTrainingInput_flabel.txt', delimiter = ',')
        print(pred_dl_input_df.corr(method = 'pearson'))
        # CALCULATE THE P-VALUE FOR TEST
        train_auc_list = list(pred_dl_input_df['AUC'])
        train_auc = np.array(train_auc_list)
        train_pred_list = list(pred_dl_input_df['Pred Score'])
        train_pred = np.array(train_pred_list)
        train_false_list = list(pred_dl_input_df['False AUC'])
        train_false = np.array(train_false_list)
        print('--- P-VALUE BETWEEN F-AUC & AUC ---')
        t1, p1 = stats.ttest_ind(train_false, train_auc)
        print(p1)
        print('--- P-VALUE BETWEEN AUC & PRED ---')
        t2, p2 = stats.ttest_ind(train_auc, train_pred)
        print(p2)

    def test_pvalue(self, path):
        # ALL POINTS PREDICTION SCATTERPLOT
        dir_opt = self.dir_opt
        pred_dl_input_df = pd.read_csv(path + '/PredTestInput_flabel.txt', delimiter = ',')
        print(pred_dl_input_df.corr(method = 'pearson'))
        # CALCULATE THE P-VALUE FOR TEST
        test_auc_list = list(pred_dl_input_df['AUC'])
        test_auc = np.array(test_auc_list)
        test_pred_list = list(pred_dl_input_df['Pred Score'])
        test_pred = np.array(test_pred_list)
        test_false_list = list(pred_dl_input_df['False AUC'])
        test_false = np.array(test_false_list)

        print('--- P-VALUE BETWEEN F-AUC & AUC ---')
        t1, p1 = stats.ttest_ind(test_false, test_auc)
        print(p1)
        print('--- P-VALUE BETWEEN AUC & PRED ---')
        t2, p2 = stats.ttest_ind(test_auc, test_pred)
        print(p2)

    def scatter_kde_train(self, path, epoch_time):
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

        # Generate random data for each categorical variable:
        x = train_auc
        y = train_pred

        # Set up 4 subplots as axis objects using GridSpec:
        gs = gridspec.GridSpec(2, 2, width_ratios=[1,3], height_ratios=[3,1])
        # Add space between scatter plot and KDE plots to accommodate axis labels:
        gs.update(hspace=0.3, wspace=0.3)

        # Set background canvas colour to White instead of grey default
        fig = plt.figure()
        fig.patch.set_facecolor('white')

        ax = plt.subplot(gs[0,1]) # Instantiate scatter plot area and axis range
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_xlabel('AUC')
        ax.set_ylabel('Pred')

        axl = plt.subplot(gs[0,0], sharey=ax) # Instantiate left KDE plot area
        axl.get_xaxis().set_visible(False) # Hide tick marks and spines
        axl.get_yaxis().set_visible(False)
        axl.spines["right"].set_visible(False)
        axl.spines["top"].set_visible(False)
        axl.spines["bottom"].set_visible(False)

        axb = plt.subplot(gs[1,1], sharex=ax) # Instantiate bottom KDE plot area
        axb.get_xaxis().set_visible(False) # Hide tick marks and spines
        axb.get_yaxis().set_visible(False)
        axb.spines["right"].set_visible(False)
        axb.spines["top"].set_visible(False)
        axb.spines["left"].set_visible(False)

        axc = plt.subplot(gs[1,0]) # Instantiate legend plot area
        axc.axis('off') # Hide tick marks and spines

        ax.scatter(x, y)

        kde = stats.gaussian_kde(x)
        xx = np.linspace(x.min(), x.max(), 1000)
        axb.plot(xx, kde(xx), color='black')

        kde = stats.gaussian_kde(y)
        yy = np.linspace(y.min(), y.max(), 1000)
        axl.plot(kde(yy), yy, color='black')

        # SAVE TRAINING PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_train_scatter_kde'
        path = '.' + dir_opt + '/plot/%s' % (file_name) + '.png'
        unit = 1
        if os.path.exists('.' + dir_opt + '/plot') == False:
            os.mkdir('.' + dir_opt + '/plot')
        while os.path.exists(path):
            path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(path, dpi = 300)


    def scatter_kde_test(self, path, epoch_time):
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

        # Generate random data for each categorical variable:
        x = test_auc
        y = test_pred

        # Set up 4 subplots as axis objects using GridSpec:
        gs = gridspec.GridSpec(2, 2, width_ratios=[1,3], height_ratios=[3,1])
        # Add space between scatter plot and KDE plots to accommodate axis labels:
        gs.update(hspace=0.3, wspace=0.3)

        # Set background canvas colour to White instead of grey default
        fig = plt.figure()
        fig.patch.set_facecolor('white')

        ax = plt.subplot(gs[0,1]) # Instantiate scatter plot area and axis range
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_xlabel('AUC')
        ax.set_ylabel('Pred')

        axl = plt.subplot(gs[0,0], sharey=ax) # Instantiate left KDE plot area
        axl.get_xaxis().set_visible(False) # Hide tick marks and spines
        axl.get_yaxis().set_visible(False)
        axl.spines["right"].set_visible(False)
        axl.spines["top"].set_visible(False)
        axl.spines["bottom"].set_visible(False)

        axb = plt.subplot(gs[1,1], sharex=ax) # Instantiate bottom KDE plot area
        axb.get_xaxis().set_visible(False) # Hide tick marks and spines
        axb.get_yaxis().set_visible(False)
        axb.spines["right"].set_visible(False)
        axb.spines["top"].set_visible(False)
        axb.spines["left"].set_visible(False)

        axc = plt.subplot(gs[1,0]) # Instantiate legend plot area
        axc.axis('off') # Hide tick marks and spines

        ax.scatter(x, y)

        kde = stats.gaussian_kde(x)
        xx = np.linspace(x.min(), x.max(), 1000)
        axb.plot(xx, kde(xx), color='black')

        kde = stats.gaussian_kde(y)
        yy = np.linspace(y.min(), y.max(), 1000)
        axl.plot(kde(yy), yy, color='black')

        # SAVE TEST PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_test_scatter_kde'
        path = '.' + dir_opt + '/plot/%s' % (file_name) + '.png'
        unit = 1
        if os.path.exists('.' + dir_opt + '/plot') == False:
            os.mkdir('.' + dir_opt + '/plot')
        while os.path.exists(path):
            path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(path, dpi = 300)


if __name__ == "__main__":
    dir_opt = '/datainfo'
    epoch_time = '100'
    path = '.' + dir_opt + '/result/5-fold-flabel01/epoch_99_4'
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm2'
    pathway_filename = 'Selected_Kegg_Pathways2'
    post_data_path = '.' + dir_opt + '/post_data' # important may change xTe, xTr


    # # ANALYSE DRUG EFFECT
    # print('ANALYSING DRUG EFFECT...')
    # Analyse(dir_opt).plot_train_real_pred(path, epoch_time)
    # Analyse(dir_opt).plot_test_real_pred(path, epoch_time)

    # ANALYSE P-VALUE
    Analyse(dir_opt).train_pvalue(path)
    Analyse(dir_opt).scatter_kde_train(path, epoch_time)
    Analyse(dir_opt).test_pvalue(path)
    Analyse(dir_opt).scatter_kde_test(path, epoch_time)