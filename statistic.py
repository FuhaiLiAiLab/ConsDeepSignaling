import os
import math
import numpy as np
import pandas as pd
from numpy import savetxt

from load_data import LoadData

class Statistic():
    def __init__(self):
        pass

    # CALCULATE THE EACH STEP [rna_fpkm, cnv_gistic] [gene, celllines]
    def gdsc_flow_shape():
        # RAW DATA [RNA_seq, CpNum] (RNA_seq[37279, 1047], CpNum[20669, 978])
        # FILTERED (1/3 values with 0) DATA [RNA_seq] (RNA_seq[9936, 1047], CpNum[20669, 978])
        # TAILED (RNA_seq, CpNum intersection) (RNA_seq[9268, 971], CpNum[9268, 971])
        # INTERSECT DL_INPUT [9268, 791]
        # INTERSECT GENE_PATHWAY [929, 791]
        tailed_rna_df = pd.read_csv('./datainfo/filtered_data/tailed_rnaseq_fpkm_20191101.csv')
        tailed_cnv_df = pd.read_csv('./datainfo/filtered_data/tailed_cnv_gistic_20191101.csv')
        print(tailed_rna_df.shape)
        print(tailed_cnv_df.shape)
    
    # CALCULATE THE CELL LINES NUMBER IN [Tailed] [929, 46]
    def tailed_gene_pathway_count():
        order_gene_pathway_df = pd.read_csv('./datainfo/mid_data/Ordered_Selected_Kegg_Pathways2.csv')
        print(order_gene_pathway_df.shape)
        tailed_gene_pathway_df = pd.read_csv('./datainfo/filtered_data/Tailed_Selected_Kegg_Pathways2.csv')
        tailed_gene_pathway_df = tailed_gene_pathway_df.drop(columns = ['Unnamed: 0'])
        print(tailed_gene_pathway_df.shape)

    # FINAL [dl_input] [16761]
    def dl_input_flow_shape(): 
        # dl_second_df = pd.read_excel('./GDSC/dose_responce_25Feb20/GDSC2_fitted_dose_response_25Feb20.xlsx')
        # print('--- DL INITIAL POINTS: ' + str(dl_second_df.shape[0]) + ' ---')
        dl_second_input_df = pd.read_csv('./datainfo/mid_data/GDSC2_dl_input.txt')
        print('--- DL INITIAL AVERAGE POINTS: ' + str(dl_second_input_df.shape[0]) + ' ---')
        mid_dl_input_df = pd.read_csv('./datainfo/mid_data/mid_GDSC2_dl_input.txt')
        print('--- DL DRUG CONDENSED INPUT POINTS: ' + str(mid_dl_input_df.shape[0]) + ' ---')
        drug_target_matrix = np.load('./datainfo/filtered_data/drug_target_matrix.npy')
        print(drug_target_matrix.shape)

    # CALCULATE THE UNIQUE DRUGS IN [zero_final_GDSC2_dl_input]
    def zero_final_drug_count():
        zero_final_dl_input_df = pd.read_table('./datainfo/filtered_data/zerofinal_GDSC2_dl_input.txt', delimiter = ',')
        zero_final_drug_list = []
        for drug in zero_final_dl_input_df['DRUG_NAME']:
            if drug not in zero_final_drug_list:
                zero_final_drug_list.append(drug)
        zero_final_drug_list = sorted(zero_final_drug_list)
        print(zero_final_drug_list)
        print(len(zero_final_drug_list))
        # Count the Number of Drugs Intersection Between [dl_input, drugBank]
        dir_opt = '/datainfo'
        drug_map_dict, drug_dict, gene_target_num_dict = LoadData(dir_opt).pre_load_dict()
        count = 0
        anti_count = 0
        for key, value in drug_map_dict.items():
            if type(value) is str:
                count = count + 1
            elif math.isnan(value) == True:
                anti_count = anti_count + 1
        print(count)
        print(anti_count)


    # CALCULATE THE UNIQUE CELLLINES IN [zero_final_GDSC2_dl_input]
    def zero_final_cellline_count():
        zero_final_dl_input_df = pd.read_table('./datainfo/filtered_data/zerofinal_GDSC2_dl_input.txt', delimiter = ',')
        zero_final_cellline_list = []
        for cellline in zero_final_dl_input_df['CELL_LINE_NAME']:
            if cellline not in zero_final_cellline_list:
                zero_final_cellline_list.append(cellline)
        zero_final_cellline_list = sorted(zero_final_cellline_list)
        print(zero_final_cellline_list)
        print(len(zero_final_cellline_list))


if __name__ == "__main__":

    # CALCULATE UNIQUE DRUG AND CELL_LINE NAMES 
    Statistic.zero_final_drug_count()
    # Statistic.zero_final_cellline_count()

    # Statistic.gdsc_flow_shape()
    # Statistic.dl_input_flow_shape()
    # Statistic.tailed_gene_pathway_count()



    