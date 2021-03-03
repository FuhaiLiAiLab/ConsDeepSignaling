import os
import numpy as np
import pandas as pd
from numpy import savetxt
from sklearn.metrics import mean_squared_error

from load_data_flabel import LoadData

class Reform():
    def __init__(self, dir_opt):
        self.dir_opt = dir_opt

    def drug_gene_reform(self):
        dir_opt = self.dir_opt
        rna_df = pd.read_csv('./datainfo/filtered_data/tailed_rnaseq_fpkm_20191101.csv')
        gene_list = list(rna_df['symbol'])
        drug_map_dict, drug_dict, gene_target_num_dict = LoadData(dir_opt).pre_load_dict()
        # 929 MAPPED GENES INDEX IN [drug_target_matrix]
        target_index_list = gene_target_num_dict.values()
        # 24 DRUGS IN DEEP LEARNING TASK
        zero_final_dl_input_df = pd.read_table('./datainfo/filtered_data/zerofinal_GDSC2_dl_input.txt', delimiter = ',')
        zero_final_drug_list = []
        for drug in zero_final_dl_input_df['DRUG_NAME']:
            if drug not in zero_final_drug_list:
                zero_final_drug_list.append(drug)
        zero_final_drug_list = sorted(zero_final_drug_list)
        # 24 MAPPED DRUGS IN DEEP LEARNING TASK
        mapped_drug_list = []
        for zero_drug in zero_final_drug_list:
            mapped_drug_list.append(drug_map_dict[zero_drug])
        drug_target_matrix = np.load('.' + dir_opt + '/filtered_data/drug_target_matrix.npy')
        # FIND DRUGS CAN TARGET ON GENES
        multi_drug_list = []
        for target_index in target_index_list:
            temp_drug_list = []
            if target_index == -1:
                temp_drug_list = ['NaN']
                multi_drug_list.append(temp_drug_list)
            else:
                for mapped_drug in mapped_drug_list:
                    drug_index = drug_dict[mapped_drug]
                    effect = drug_target_matrix[drug_index, target_index]
                    if effect == 1: temp_drug_list.append(mapped_drug)
                if len(temp_drug_list) == 0: temp_drug_list = ['NaN']
                multi_drug_list.append(temp_drug_list)
        print(multi_drug_list)
        print(len(multi_drug_list))
        d = {'Drugs': multi_drug_list, 'Genes': gene_list}
        df = pd.DataFrame(d, columns=['Drugs','Genes'])
        df.to_csv('./datainfo/filtered_data/drug_gene.csv', index = False, header = True)
        a_df = pd.read_csv('./datainfo/filtered_data/drug_gene.csv')
        print(a_df)


    def drug_gene_pathway_reform():
        return 0


if __name__ == "__main__":
    dir_opt = '/datainfo'

    Reform(dir_opt).drug_gene_reform()