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

        # import pdb; pdb.set_trace()
        if os.path.exists('./datainfo/analysis_data') == False:
            os.mkdir('./datainfo/analysis_data')
        file_name = 'analysis_' + epoch_time + '_parameter'
        analysis_param_path = '.' + dir_opt + '/analysis_data/%s' % (file_name) + '.npy'
        unit = 1
        while os.path.exists(analysis_param_path):
            analysis_param_path = '.' + dir_opt + '/analysis_data/%s_%d' % (file_name, unit) + '.npy'
            unit += 1
        np.save(analysis_param_path, analysis)

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
    

    def combine_analysis(self, RNA_seq_filename, pathway_filename, path, post_data_path, epoch_time):
        file_name = 'analysis_' + epoch_time + '_parameter'
        analysis_param_path = '.' + dir_opt + '/analysis_data/%s' % (file_name) + '.npy'
        name_list = []
        name_list.append(analysis_param_path)
        for unit in range(1, 5):
            analysis_param_path = '.' + dir_opt + '/analysis_data/%s_%d' % (file_name, unit) + '.npy'
            name_list.append(analysis_param_path)
        num_pathway = np.load(name_list[0]).shape[1]
        combine_analysis = np.zeros((1, num_pathway))
        for name in name_list:
            temp_analysis = np.load(name)
            combine_analysis = np.vstack((combine_analysis, temp_analysis))
        combine_analysis = np.delete(combine_analysis, 0, axis = 0)
        # import pdb; pdb.set_trace()
        # PATHWAY DISTRIBUTION ANALYSIS ON TEST INDEX
        print('COMBINED PATHWAY DISTRIBUTION ANALYSIS...')
        file_name = 'combined_epoch_' + epoch_time + '_pathway_distribution'
        distribution_path = '.' + dir_opt + '/plot/%s' % (file_name) + '.pdf'
        unit = 1
        while os.path.exists(distribution_path):
            distribution_path = '.' + dir_opt + '/plot/%s_%d' % (file_name, unit) + '.pdf'
            unit += 1
        pdf = PdfPages(distribution_path)
        gene_pathway_df = pd.read_table('.' + dir_opt + '/init_data/' + pathway_filename + '.txt')
        pathway_name_list = list(gene_pathway_df.columns[1:])
        pathway_name = pathway_name_list
        fig = plt.figure(figsize = (15, 20))
        for i in range(num_pathway):
            ax = fig.add_subplot(11, 5, i+1)
            plt.xlim([-0.1, 0.1])
            sns.kdeplot(combine_analysis[:,i], shade = True)
            plt.title( pathway_name[i],fontsize = 'small', fontweight='bold')
            plt.tight_layout()
            plt.subplots_adjust(wspace = 0.5, hspace = 0.5)
        pdf.savefig()
        plt.close()
        pdf.close()

               



if __name__ == "__main__":
    dir_opt = '/datainfo'
    epoch_time = '100'
    path = '.' + dir_opt + '/result/5-fold/epoch_99_No_4'
    RNA_seq_filename = 'nci60-ccle_RNAseq_tpm2'
    pathway_filename = 'Selected_Kegg_Pathways2'
    post_data_path = '.' + dir_opt + '/post_data' # important may change xTe, xTr


    # # ANALYSE GENES PATHWAYS
    # print('ANALYSING PATHWAY GENE EFFECT...')
    # Analyse(dir_opt).pathway_analysis(RNA_seq_filename, pathway_filename, path, post_data_path, epoch_time)
    Analyse(dir_opt).combine_analysis(RNA_seq_filename, pathway_filename, path, post_data_path, epoch_time)