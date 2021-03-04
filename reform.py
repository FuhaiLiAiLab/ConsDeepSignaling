import os
import numpy as np
import pandas as pd
from numpy import savetxt
from sklearn.metrics import mean_squared_error

from load_data import LoadData

class Reform():
    def __init__(self, dir_opt):
        self.dir_opt = dir_opt

    def drug_gene_pathway_reform(self):
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
        # CONVERT EACH GENES TARGETED DRUGS TO DATAFRAME
        drug_gene = {'Drugs': multi_drug_list, 'Genes': gene_list}
        drug_gene_df = pd.DataFrame(drug_gene, columns=['Drugs','Genes'])
        drug_gene_df.to_csv('./datainfo/filtered_data/drug_gene.csv', index = False, header = True)

        # ADD PATHWAYS TO CORRESPONDING GENES
        gene_pathway_df = pd.read_csv('./datainfo/filtered_data/Tailed_Selected_Kegg_Pathways2.csv')
        pathway_name_list = list(gene_pathway_df.columns)[1:]
        multi_pathway_list = []
        # import pdb; pdb.set_trace()
        for row in gene_pathway_df.itertuples():
            temp_pathway_list = []
            for index in np.arange(2, 48):
                if row[index] == 1: 
                    temp_pathway_list.append(pathway_name_list[index - 2])
            if len(temp_pathway_list) == 0:
                temp_pathway_list = ['NaN']
                print(row[1])
            multi_pathway_list.append(temp_pathway_list)
        # print(multi_pathway_list)
        # print(len(multi_pathway_list))
        # CONVERT EACH GENES TARGETED DRUGS/ CONNECTION TO PATHWAYS TO DATAFRAME
        drug_gene_pathway = {'Drugs': multi_drug_list, 'Genes': gene_list, 'Pathways': multi_pathway_list}
        drug_gene_pathway_df = pd.DataFrame(drug_gene_pathway, columns=['Drugs', 'Genes', 'Pathways'])
        drug_gene_pathway_df.to_csv('./datainfo/filtered_data/drug_gene_pathway.csv', index = False, header = True)
        print(drug_gene_pathway_df)


if __name__ == "__main__":
    dir_opt = '/datainfo'
    Reform(dir_opt).drug_gene_pathway_reform()