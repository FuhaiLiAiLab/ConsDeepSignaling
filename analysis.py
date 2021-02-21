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
        print(pred_dl_input_df.corr(method = 'pearson'))
        # CALCULATE THE MSE FOR TRAIN
        train_auc_list = list(pred_dl_input_df['AUC'])
        train_auc = np.array(train_auc_list)
        train_pred_list = list(pred_dl_input_df['Pred Score'])
        train_pred = np.array(train_pred_list)
        train_mse = mean_squared_error(train_auc, train_pred)
        print('Training MSE: ' + str(train_mse))
        print(pred_dl_input_df.corr(method = 'pearson'))
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
        pred_dl_input_df = pd.read_csv(path + '/PredTestInput.txt', delimiter = ',')
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
    

    def gene_analysis(self, RNA_seq_filename, pathway_filename, path, post_data_path, epoch_time):
        # ANALYSE PATHWAY GENE AND DISTRIBUTION
        dir_opt = self.dir_opt
        zero_final_dl_input_df, input_num, num_feature, cellline_gene_df, num_gene, num_pathway = LoadData(dir_opt, RNA_seq_filename).pre_load_train()
        layer0 = num_pathway
        layer1 = 256
        layer2 = 128
        layer3 = 32
        matrixA = GenMatrix(dir_opt, RNA_seq_filename, pathway_filename).feature_gene_matrix(num_feature, num_gene)
        matrixB = GenMatrix(dir_opt, RNA_seq_filename, pathway_filename).gene_pathway_matrix(num_pathway)
        # RECONSTRUCT THE DECOMPOSED MODEL
        input_model, gene_model, pathway_model, model = RandNN().keras_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3)
        # AUTO REBUILD THE MODEL
        # model = model.load_weights(path + '/model.h5')
        # MANUAL REBUILD THE MODEL (BETTER)
        with open(path + '/layer_list.txt', 'rb') as filelayer:
            layer_list = pickle.load(filelayer)
        model.compile(loss='mean_squared_error',
                        optimizer='adam',
                        metrics=['mse', 'accuracy']) 
        xTmp, yTmp = LoadData(dir_opt, RNA_seq_filename).load_train(0, 1)
        model.fit(xTmp, yTmp, epochs = 1, validation_split = 1, verbose = 0)
        num_layer = len(model.layers)
        for i in range(num_layer):
            model.get_layer(index = i).set_weights(layer_list[i])
        # GENE IMPORTANT ANALYSIS
        # ON TRAINING DATA SET
        xTr = np.load(post_data_path + '/xTr.npy')
        gene_x = np.array(input_model.predict(xTr))
        pathway_analyzer = innvestigate.create_analyzer('smoothgrad', gene_model, noise_scale=(np.max(gene_x)-np.min(gene_x))*0.1)
        analysis = pathway_analyzer.analyze(gene_x)
        mean_analysis = np.mean(analysis, axis = 0)
        print('Gene Analysis')
        print(mean_analysis.shape)
        # MAKE BARPOLT ON GENES IMPORTANCE
        top = 50
        mean_analysis = np.absolute(mean_analysis)
        # GET AND SAVE TOP GENE INDEX
        top_gene_index = heapq.nlargest(top, range(len(mean_analysis)), mean_analysis.take)
        np.save(path + '/train_top_gene_index' + epoch_time + '.npy', top_gene_index)
        top_gene_value = mean_analysis[top_gene_index]
        cellline_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv')
        gene_name_list = cellline_df['geneSymbol']
        gene_name_dict = {k : v for k, v in enumerate(gene_name_list)}
        top_gene_name = [gene_name_dict.get(key) for key in top_gene_index]
        top_gene_name_value_dict = dict(zip(top_gene_name, top_gene_value))
        # BARPLOT ON GENES AND SAVE
        plt.figure(figsize = (16, 9))
        plt.bar(range(len(top_gene_name_value_dict)), list(top_gene_name_value_dict.values()))
        plt.xticks(range(len(top_gene_name_value_dict)), list(top_gene_name_value_dict.keys()),
                    rotation = 30, fontsize = 10, ha = 'right')
        plt.title('Top 50 Genes With Largest Absolute Importance Scores of 1684 Genes On Training Dataset',
                    fontsize = 16)
        plt.tight_layout()
        # SAVE TRAINING PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_train_gene_barplot'
        train_path = '.' + dir_opt + '/plot/%s' % (file_name) + '.png'
        unit = 1
        while os.path.exists(train_path):
            train_path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(train_path, dpi = 300)
        # ON TEST DATA SET
        xTe = np.load(post_data_path + '/xTe.npy')
        gene_x = np.array(input_model.predict(xTe))
        gene_analyzer = innvestigate.create_analyzer('smoothgrad', gene_model, noise_scale=(np.max(gene_x)-np.min(gene_x))*0.1)
        analysis = gene_analyzer.analyze(gene_x)
        mean_analysis = np.mean(analysis, axis = 0)
        print('Gene Analysis')
        print(mean_analysis.shape)
        # MAKE BARPOLT ON GENES IMPORTANCE
        top = 50
        mean_analysis = np.absolute(mean_analysis)
        # GET AND SAVE TOP GENE INDEX
        top_gene_index = heapq.nlargest(top, range(len(mean_analysis)), mean_analysis.take)
        np.save(path + '/test_top_gene_index' + epoch_time + '.npy', top_gene_index)
        top_gene_value = mean_analysis[top_gene_index]
        cellline_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv')
        gene_name_list = cellline_df['geneSymbol']
        gene_name_dict = {k : v for k, v in enumerate(gene_name_list)}
        top_gene_name = [gene_name_dict.get(key) for key in top_gene_index]
        top_gene_name_value_dict = dict(zip(top_gene_name, top_gene_value))
        # BARPLOT ON GENES AND SAVE
        plt.figure(figsize = (16, 9))
        plt.bar(range(len(top_gene_name_value_dict)), list(top_gene_name_value_dict.values()))
        plt.xticks(range(len(top_gene_name_value_dict)), list(top_gene_name_value_dict.keys()),
                    rotation = 30, fontsize = 10, ha = 'right')
        plt.title('Top 50 Genes With Largest Absolute Importance Scores of 1684 Genes On Test Dataset',
                    fontsize = 16)
        plt.tight_layout()
        # SAVE TEST PLOT FIGURE
        file_name = 'epoch_' + epoch_time + '_test_gene_barplot'
        test_path = '.' + dir_opt + '/plot/%s' % (file_name) + '.png'
        unit = 1
        while os.path.exists(test_path):
            test_path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.png'
            unit += 1
        plt.savefig(test_path, dpi = 300)
        


    def pathway_analysis(self, RNA_seq_filename, pathway_filename, path, post_data_path, epoch_time):
        # ANALYSE PATHWAY GENE AND DISTRIBUTION
        dir_opt = self.dir_opt
        train_input_df, input_num, num_feature, rna_df, cpnum_df, num_gene, num_pathway = LoadData(dir_opt).pre_load_train()
        layer0 = num_pathway
        layer1 = 256
        layer2 = 128
        layer3 = 32
        matrixA = GenMatrix.feature_gene_matrix(num_feature, num_gene)
        matrixB = GenMatrix.gene_pathway_matrix()
        # RECONSTRUCT THE DECOMPOSED MODEL
        input_model, gene_model, pathway_model, model = RandNN().keras_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3)
        # AUTO REBUILD THE MODEL
        # model = model.load_weights(path + '/model.h5')
        # MANUAL REBUILD THE MODEL (BETTER)
        with open(path + '/layer_list.txt', 'rb') as filelayer:
            layer_list = pickle.load(filelayer)
        model.compile(loss='mean_squared_error',
                        optimizer='adam',
                        metrics=['mse', 'accuracy']) 
        xTmp, yTmp = LoadData(dir_opt).load_train(0, 1)
        model.fit(xTmp, yTmp, epochs = 1, validation_split = 1, verbose = 0)
        num_layer = len(model.layers)
        for i in range(num_layer):
            model.get_layer(index = i).set_weights(layer_list[i])
        # RE-LOAD EACH MODEL
        input_model = model.layers[1]
        gene_model = model.layers[2]
        pathway_model = model.layers[3]
        # GENE IMPORTANT ANALYSIS
        # ON TEST DATA ONLY
        xTe = np.load(post_data_path + '/xTe.npy')
        gene_x = np.array(input_model.predict(xTe))
        # PATHWAY INPORTANCE ANALYSIS ON TEST INDEX
        # import pdb; pdb.set_trace()
        pathway_x = gene_model.predict(gene_x)
        gene_pathway_df = pd.read_table('.' + dir_opt + '/init_data/' + pathway_filename + '.txt')
        pathway_name_list = list(gene_pathway_df.columns[1:])
        pathway_analyzer = innvestigate.create_analyzer("smoothgrad", pathway_model,
                                                        noise_scale=(np.max(pathway_x)-np.min(pathway_x))*0.1)
        analysis = pathway_analyzer.analyze(pathway_x)

        print('PATHWAY IMPORTANCE ANALYSIS...')
        file_name = 'epoch_' + epoch_time + '_pathway_analysis'
        analysis_path = '.' + dir_opt + '/plot/%s' % (file_name) + '.pdf'
        unit = 1
        while os.path.exists(analysis_path):
            analysis_path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.pdf'
            unit += 1
        pdf = PdfPages(analysis_path)
        plt.figure(figsize = (8, 6))
        ax = plt.imshow(analysis.squeeze(), cmap='seismic', interpolation='nearest', aspect="auto")
        cb = plt.colorbar(ax)
        cb.ax.tick_params(labelsize = 8)
        plt.ylabel('Sample Index')
        plt.xlabel('Pathway')
        plt.xticks(rotation = 45)
        plt.tick_params(labelsize = 8)
        pdf.savefig()
        plt.close()
        pdf.close()

        # PATHWAY DISTRIBUTION ANALYSIS ON TEST INDEX
        print('PATHWAY DISTRIBUTION ANALYSIS...')
        file_name = 'epoch_' + epoch_time + '_pathway_distribution'
        distribution_path = '.' + dir_opt + '/plot/%s' % (file_name) + '.pdf'
        unit = 1
        while os.path.exists(distribution_path):
            distribution_path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.pdf'
            unit += 1
        pdf = PdfPages(distribution_path)
        pathway_name = pathway_name_list
        fig = plt.figure(figsize = (15, 20))
        for i in range(num_pathway):
            ax = fig.add_subplot(11, 5, i+1)
            plt.xlim([-0.1, 0.1])
            sns.kdeplot(analysis[:,i], shade = True)
            plt.title( pathway_name[i],fontsize = 'small', fontweight='bold')
            plt.tight_layout()
            plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
        pdf.savefig()
        plt.close()
        pdf.close()


    def gene_intersect(self, epoch_time):
        # TOP INTERSECT GENES
        cellline_df = pd.read_csv('.' + dir_opt + '/filtered_data/' + RNA_seq_filename + '.csv')
        gene_name_list = cellline_df['geneSymbol']
        gene_name_dict = {k : v for k, v in enumerate(gene_name_list)}
        # LOAD GENE INDICES FROM FILES
        path = '.' + dir_opt + '/result/epoch_30'
        train_top_gene_index = np.load(path + '/train_top_gene_index' + epoch_time + '.npy')
        test_top_gene_index = np.load(path + '/test_top_gene_index' + epoch_time + '.npy')
        path = '.' + dir_opt + '/result/epoch_30_1'
        train_top_gene_index1 = np.load(path + '/train_top_gene_index' + epoch_time + '.npy')
        test_top_gene_index1 = np.load(path + '/test_top_gene_index' + epoch_time + '.npy')
        path = '.' + dir_opt + '/result/epoch_30_2'
        train_top_gene_index2 = np.load(path + '/train_top_gene_index' + epoch_time + '.npy')
        test_top_gene_index2 = np.load(path + '/test_top_gene_index' + epoch_time + '.npy')
        path = '.' + dir_opt + '/result/epoch_30_3'
        train_top_gene_index3 = np.load(path + '/train_top_gene_index' + epoch_time + '.npy')
        test_top_gene_index3 = np.load(path + '/test_top_gene_index' + epoch_time + '.npy')
        path = '.' + dir_opt + '/result/epoch_30_4'
        train_top_gene_index4 = np.load(path + '/train_top_gene_index' + epoch_time + '.npy')
        test_top_gene_index4 = np.load(path + '/test_top_gene_index' + epoch_time + '.npy')
        # INTERSECT ON TRAIN
        train_top_gene_set = set(train_top_gene_index)
        train_top_gene_set1 = set(train_top_gene_index1)
        train_top_gene_set2 = set(train_top_gene_index2)
        train_top_gene_set3 = set(train_top_gene_index3)
        train_top_gene_set4 = set(train_top_gene_index4)
        set1 = train_top_gene_set.intersection(train_top_gene_set1)
        set2 = set1.intersection(train_top_gene_index2)
        set3 = set2.intersection(train_top_gene_index3)
        train_result_set = set3.intersection(train_top_gene_index4)
        train_result_list = list(train_result_set)
        train_result_gene_name = [gene_name_dict.get(key) for key in train_result_list]
        print('train_result_gene_name: ', train_result_gene_name)
        # INTERSECT ON TEST
        test_top_gene_set = set(test_top_gene_index)
        test_top_gene_set1 = set(test_top_gene_index1)
        test_top_gene_set2 = set(test_top_gene_index2)
        test_top_gene_set3 = set(test_top_gene_index3)
        test_top_gene_set4 = set(test_top_gene_index4)
        set1 = test_top_gene_set.intersection(test_top_gene_set1)
        set2 = set1.intersection(test_top_gene_index2)
        set3 = set2.intersection(test_top_gene_index3)
        test_result_set = set3.intersection(test_top_gene_index4)
        test_result_list = list(test_result_set)
        test_result_gene_name = [gene_name_dict.get(key) for key in test_result_list]
        print('test_result_gene_name: ', test_result_gene_name)
        # INTERSECT ON ALL
        all_result_set = train_result_set.intersection(test_result_set)
        all_result_list = list(all_result_set)
        all_result_gene_name = [gene_name_dict.get(key) for key in all_result_list]
        print('all_result_gene_name: ', all_result_gene_name)
    



if __name__ == "__main__":
    dir_opt = '/datainfo'
    epoch_time = '100'
    path = '.' + dir_opt + '/result/3-fold/epoch_99_No_2'
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm2'
    pathway_filename = 'Selected_Kegg_Pathways2'
    post_data_path = '.' + dir_opt + '/post_data' # important may change xTe, xTr


    # ANALYSE DRUG EFFECT
    print('ANALYSING DRUG EFFECT...')
    Analyse(dir_opt).plot_train_real_pred(path, epoch_time)
    Analyse(dir_opt).plot_test_real_pred(path, epoch_time)

    # # ANALYSE GENES PATHWAYS
    # print('ANALYSING PATHWAY GENE EFFECT...')
    # # Analyse(dir_opt).gene_analysis(RNA_seq_filename, pathway_filename, path, post_data_path, epoch_time)
    Analyse(dir_opt).pathway_analysis(RNA_seq_filename, pathway_filename, path, post_data_path, epoch_time)
    # Analyse(dir_opt).gene_intersect(epoch_time)