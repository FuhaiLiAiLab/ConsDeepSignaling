import os
import numpy as np
import pandas as pd
from numpy import savetxt

from load_data import LoadData

class ParseFile():
    def __init__(self):
        pass

    # FIND THE DUPLICATE ROWS[CELL_LINE_NAME, DRUG_NAME, AUC] THEN AVERAGE SCORE
    def second_input_condense():
        print('\nREADING THE EXCEL FILE FOR DEEP LEARNING DOSE RESPONSE...')
        dl_second_df = pd.read_excel('./GDSC/dose_responce_25Feb20/GDSC2_fitted_dose_response_25Feb20.xlsx')
        dl_second_input_df = dl_second_df[['CELL_LINE_NAME', 'DRUG_NAME', 'AUC']]
        dl_second_input_df = dl_second_input_df.groupby(['CELL_LINE_NAME', 'DRUG_NAME']).agg({'AUC':'mean'}).reset_index()
        if os.path.exists('./datainfo/mid_data') == False:
            os.mkdir('./datainfo/mid_data')
        dl_second_input_df.to_csv('./datainfo/mid_data/GDSC2_dl_input.txt', index = False, header = True)
        print('--- DL INITIAL AVERAGE POINTS: ' + str(dl_second_input_df.shape[0]) + ' ---')

    # REMOVE INPUT ROWS WITH NO MAPPED DRUG NAME (FINALLY 39066 POINTS INPUT)
    def input_drug_condense():
        dl_input_df = pd.read_table('./datainfo/mid_data/GDSC2_dl_input.txt', delimiter = ',')
        drug_map_dict = ParseFile.drug_map_dict()
        deletion_list = []
        for row in dl_input_df.itertuples():
            if pd.isna(drug_map_dict[row[2]]):
                deletion_list.append(row[0])
        mid_dl_input_df = dl_input_df.drop(dl_input_df.index[deletion_list]).reset_index(drop = True)
        mid_dl_input_df.to_csv('./datainfo/mid_data/mid_GDSC2_dl_input.txt', index = False, header = True)
        print('--- DL DRUG CONDENSED INPUT POINTS: ' + str(mid_dl_input_df.shape[0]) + ' ---')

    # REMOVE INPUT ROWS WITH ALL ZEROS ON DRUG TARGET GENE CONNECTION, [FINAL 16761 POINTS]
    def input_drug_gene_condense():
        dir_opt = '/datainfo'
        deletion_list = []
        final_dl_input_df = pd.read_table('./datainfo/mid_data/final_GDSC2_dl_input.txt', delimiter = ',')
        drug_map_dict, drug_dict, gene_target_num_dict = LoadData(dir_opt).pre_load_dict()
        target_index_list = gene_target_num_dict.values()
        drug_target_matrix = np.load('./datainfo/filtered_data/drug_target_matrix.npy')
        for row in final_dl_input_df.itertuples():
            drug_a = drug_map_dict[row[2]]
            cellline_name = row[1]
            # DRUG_A AND 929 TARGET GENES
            drug_a_target_list = []
            drug_index = drug_dict[drug_a]
            for target_index in target_index_list:
                if target_index == -1 : 
                    effect = 0
                else:
                    effect = drug_target_matrix[drug_index, target_index]
                drug_a_target_list.append(effect)
            if all([a == 0 for a in drug_a_target_list]): 
                deletion_list.append(row[0])
        print('=====================' + str(len(deletion_list)))
        zero_final_dl_input_df = final_dl_input_df.drop(final_dl_input_df.index[deletion_list]).reset_index(drop = True)
        zero_final_dl_input_df.to_csv('.' + dir_opt + '/filtered_data/zerofinal_GDSC2_dl_input.txt', index = False, header = True)
        print(zero_final_dl_input_df)

    # RANDOMIZE THE DL INPUT
    def input_random_condense():
        zero_final_dl_input_df = pd.read_table('./datainfo/filtered_data/zerofinal_GDSC2_dl_input.txt', delimiter = ',')
        random_final_dl_input_df = zero_final_dl_input_df.sample(frac = 1)
        random_final_dl_input_df.to_csv('./datainfo/filtered_data/Randomfinal_GDSC2_dl_input.txt', index = False, header = True)
        print(random_final_dl_input_df)

    # LABEL RANDOMIZED 
    def random_label():
        random_final_dl_input_df = pd.read_table('./datainfo/filtered_data/Randomfinal_GDSC2_dl_input.txt', delimiter = ',')
        false_auc = random_final_dl_input_df['AUC'].sample(frac = 1).values
        random_final_dl_input_df.insert(2, 'False AUC', false_auc, True)
        # random_final_dl_input_df = random_final_dl_input_df.drop(columns=['AUC'])
        print(random_final_dl_input_df)
        random_final_dl_input_df.to_csv('./datainfo/filtered_data/Randomfinal_GDSC2_dl_input_flabel.txt', index = False, header = True)

    # SPLIT DEEP LEARNING INPUT INTO TRAINING AND TEST
    def split_k_fold(k, place_num):
        random_final_dl_input_df = pd.read_table('./datainfo/filtered_data/Randomfinal_GDSC2_dl_input_flabel.txt', delimiter = ',')
        print(random_final_dl_input_df)
        num_points = random_final_dl_input_df.shape[0]

        num_div = int(num_points / k)
        num_div_list = [i * num_div for i in range(0, k)]
        num_div_list.append(num_points)
        low_idx = num_div_list[place_num - 1]
        high_idx = num_div_list[place_num]
        print('\n--------TRAIN-TEST SPLIT WITH TEST FROM ' + str(low_idx) + ' TO ' + str(high_idx) + '--------')
        train_input_df = random_final_dl_input_df.drop(random_final_dl_input_df.index[low_idx : high_idx])
        print(train_input_df)
        test_input_df = random_final_dl_input_df[low_idx : high_idx]
        print(test_input_df)
        train_input_df.to_csv('./datainfo/filtered_data/TrainingInput_flabel.txt', index = False, header = True)
        test_input_df.to_csv('./datainfo/filtered_data/TestInput_flabel.txt', index = False, header = True)


    #########################################################

    # FORM DRUGS MAP BETWEEN [GDSC2_dl_input / drug_tar_bank]
    def drug_map():
        dl_input_df = pd.read_table('./datainfo/mid_data/GDSC2_dl_input.txt', delimiter = ',')
        drug_target_df = pd.read_table('./datainfo/init_data/drug_tar_drugBank_all.txt')
        drug_list = []
        for drug in dl_input_df['DRUG_NAME']:
            if drug not in drug_list:
                drug_list.append(drug)
        drug_list = sorted(drug_list)
        drug_df = pd.DataFrame(data = drug_list, columns = ['Drug Name'])
        drug_df.to_csv('./datainfo/init_data/GDSC2_input_drug_name.txt', index = False, header = True)
        mapped_drug_list = []
        for drug in drug_target_df['Drug']:
            if drug not in mapped_drug_list:
                mapped_drug_list.append(drug)
        mapped_drug_list = sorted(mapped_drug_list)
        mapped_drug_df = pd.DataFrame(data = mapped_drug_list, columns = ['Mapped Drug Name'])
        mapped_drug_df.to_csv('./datainfo/init_data/mapped_drug_name.txt', index = False, header = True)
        # LEFT JOIN TWO DATAFRAME
        drug_map_df = pd.merge(drug_df, mapped_drug_df, how='left', left_on = 'Drug Name', right_on = 'Mapped Drug Name')
        drug_map_df.to_csv('./datainfo/init_data/drug_map.csv', index = False, header = True)
        # AFTER AUTO MAP -> MANUAL MAP
    
    # FROM MANUAL MAP TO DRUG MAP DICT
    def drug_map_dict():
        drug_map_df = pd.read_csv('./datainfo/mid_data/drug_map.csv')
        drug_map_dict = {}
        for row in drug_map_df.itertuples():
            drug_map_dict[row[1]] = row[2]
        if os.path.exists('./datainfo/filtered_data') == False:
            os.mkdir('./datainfo/filtered_data')
        np.save('./datainfo/filtered_data/drug_map_dict.npy', drug_map_dict)
        return drug_map_dict

    # FORM ADAJACENT MATRIX (DRUG x TARGET) (LIST -> SORTED -> DICT -> MATRIX) (ALL 5435 DRUGS <-> ALL 2775 GENES)
    def drug_target():
        drug_target_df = pd.read_table('./datainfo/init_data/drug_tar_drugBank_all.txt')
        # GET UNIQUE SORTED DRUGLIST AND TARGET(GENE) LIST
        drug_list = []
        for drug in drug_target_df['Drug']:
            if drug not in drug_list:
                drug_list.append(drug)
        drug_list = sorted(drug_list)
        target_list = []
        for target in drug_target_df['Target']:
            if target not in target_list:
                target_list.append(target)
        target_list = sorted(target_list)
        # CONVERT THE SORTED LIST TO DICT WITH VALUE OF INDEX
        drug_dict = {drug_list[i] : i for i in range((len(drug_list)))} 
        drug_num_dict = {i : drug_list[i] for i in range((len(drug_list)))} 
        target_dict = {target_list[i] : i for i in range(len(target_list))}
        target_num_dict = {i : target_list[i] for i in range(len(target_list))}
        # ITERATE THE DATAFRAME TO DEFINE CONNETIONS BETWEEN DRUG AND TARGET(GENE)
        drug_target_matrix = np.zeros((len(drug_list), len(target_list))).astype(int)
        for index, drug_target in drug_target_df.iterrows():
            # BUILD ADJACENT MATRIX
            drug_target_matrix[drug_dict[drug_target['Drug']], target_dict[drug_target['Target']]] = 1
        drug_target_matrix = drug_target_matrix.astype(int)
        np.save('./datainfo/filtered_data/drug_target_matrix.npy', drug_target_matrix)
        # np.savetxt("drug_target_matrix.csv", drug_target_matrix, delimiter=',')
        # x, y = drug_target_matrix.shape
        # for i in range(x):
        #     # FIND DRUG TARGET OVER 100 GENES
        #     row = drug_target_matrix[i, :]
        #     if len(row[row>=1]) >= 100: print(drug_num_dict[i])
        np.save('./datainfo/filtered_data/drug_dict.npy', drug_dict)
        np.save('./datainfo/filtered_data/drug_num_dict.npy', drug_num_dict)
        np.save('./datainfo/filtered_data/target_dict.npy', target_dict)
        np.save('./datainfo/filtered_data/target_num_dict.npy', target_num_dict)
        return drug_dict, drug_num_dict, target_dict, target_num_dict

    # FILTER [RNA_Seq / CopyNumber] SPARSE GENES
    def rna_cpnum_filter(rna_filter, cpnum_filter):
        print('\nFILTERING SPARSE FEATURE GENES OF RNA_Seq & CpNum...')
        rna_df = pd.read_csv('./GDSC/rnaseq_20191101/rnaseq_fpkm_20191101.csv', low_memory = False)
        # rna_df = rna_df.fillna(0.0)
        cpnum_df = pd.read_csv('./GDSC/cnv_20191101/cnv_gistic_20191101.csv', low_memory = False)
        # RNA_Seq FILTER GENES
        if rna_filter == True:
            print(rna_df.shape)
            rna_df = rna_df.drop_duplicates(subset = ['symbol'], 
                        keep = 'first').sort_values(by = ['symbol']).reset_index(drop = True)
            threshold = int((len(rna_df.columns) - 3) / 3)
            deletion_list = []
            for row in rna_df.itertuples():
                if list(row[3:]).count(0) > threshold: 
                    deletion_list.append(row[0])
            rna_df = rna_df.drop(rna_df.index[deletion_list]).reset_index(drop = True)
            rna_df.to_csv('./GDSC/rnaseq_20191101/filtered_rnaseq_fpkm_20191101.csv', index = False, header = True)
            print(rna_df.shape)
        # CopyNumber FILTER GENES
        if cpnum_filter == True:
            print(cpnum_df.shape)
            cpnum_df = cpnum_df.drop_duplicates(subset = ['symbol'], 
                        keep = 'first').sort_values(by = ['symbol']).reset_index(drop = True)
            threshold = int((len(cpnum_df.columns) - 3) / 3)
            deletion_list = []
            for row in cpnum_df.itertuples():
                if list(row[3:]).count(0) > threshold: 
                    deletion_list.append(row[0])
            cpnum_df = cpnum_df.drop(cpnum_df.index[deletion_list]).reset_index(drop = True)
            cpnum_df.to_csv('./GDSC/cnv_20191101/filtered_cnv_gistic_20191101.csv', index = False, header = True)
            print(cpnum_df.shape)


    # GET [RNA_Seq / CopyNumber] CELL LINES NAMES & GENES, FINALLY [9268  / 971 CELLLINES]
    def rna_cpnum_intersect(rna_filter, cpnum_filter):
        print('\nFINDING INTERSECTION OF RNA_Seq & CpNum...')
        if rna_filter == True:
            rna_df = pd.read_csv('./GDSC/rnaseq_20191101/filtered_rnaseq_fpkm_20191101.csv', low_memory = False)
        else:
            rna_df = pd.read_csv('./GDSC/rnaseq_20191101/rnaseq_fpkm_20191101.csv', low_memory = False)
        rna_cellline_list = list(rna_df.columns)
        rna_gene_list = list(rna_df['symbol'])
        if cpnum_filter == True:
            cpnum_df = pd.read_csv('./GDSC/cnv_20191101/filtered_cnv_gistic_20191101.csv', low_memory = False)
        else:
            cpnum_df = pd.read_csv('./GDSC/cnv_20191101/cnv_gistic_20191101.csv', low_memory = False)
        cpnum_cellline_list = list(cpnum_df.columns)
        cpnum_gene_list = list(cpnum_df['symbol']) 
        # GET INTERSECTION OF [RNA_Seq / CopyNumber]
        rna_cellline_set = set(rna_cellline_list)
        cpnum_cellline_set = set(cpnum_cellline_list)
        common_cellline_list = list(rna_cellline_set.intersection(cpnum_cellline_set))
        rna_gene_set = set(rna_gene_list)
        cpnum_gene_set = set(cpnum_gene_list)
        common_gene_list = list(rna_gene_set.intersection(cpnum_gene_set))
        # DELETE [Cell Lines / Genes] NOT IN RNA_Seq
        rna_cellline_deletion_list = []
        for cellline in rna_cellline_list:
            if cellline not in common_cellline_list:
                rna_cellline_deletion_list.append(cellline)
        tail_cellline_rna_df = rna_df.drop(columns = rna_cellline_deletion_list)
        rna_gene_deletion_list = []
        for gene in rna_gene_list:
            if gene not in common_gene_list:
                rna_gene_deletion_list.append(gene)
        rna_gene_deletion_index = []
        for row in tail_cellline_rna_df.itertuples():
            if row[2] in rna_gene_deletion_list:
                rna_gene_deletion_index.append(row[0])
        tailed_rna_df = tail_cellline_rna_df.drop(rna_gene_deletion_index).reset_index(drop = True)
        tailed_sort_rna_df = tailed_rna_df.sort_values(by = ['symbol'])
        tailed_sort_rna_df.to_csv('./GDSC/rnaseq_20191101/tailed_rnaseq_fpkm_20191101.csv', index = False, header = True)
        # print(tailed_sort_rna_df)
        # DELETE [Cell Lines / Genes] NOT IN CopyNumber
        cpnum_cellline_deletion_list = []
        for cellline in cpnum_cellline_list:
            if cellline not in common_cellline_list:
                cpnum_cellline_deletion_list.append(cellline)
        tail_cellline_cpnum_df = cpnum_df.drop(columns = cpnum_cellline_deletion_list)
        cpnum_gene_deletion_list = []
        for gene in cpnum_gene_list:
            if gene not in common_gene_list:
                cpnum_gene_deletion_list.append(gene)
        cpnum_gene_deletion_index = []
        for row in tail_cellline_cpnum_df.itertuples():
            if row[2] in cpnum_gene_deletion_list:
                cpnum_gene_deletion_index.append(row[0])
        tailed_cpnum_df = tail_cellline_cpnum_df.drop(cpnum_gene_deletion_index).reset_index(drop = True)
        tailed_sort_cpnum_df = tailed_cpnum_df.sort_values(by = ['symbol'])
        tailed_sort_cpnum_df.to_csv('./GDSC/cnv_20191101/tailed_cnv_gistic_20191101.csv', index = False, header = True)
        # print(tailed_sort_cpnum_df)
        # CONFIRMATION ON TAILED FILES' [GENES, CELLLINES] ORDER
        tailed_rna_cellline_list = list(tailed_sort_rna_df.columns)
        tailed_rna_gene_list = list(tailed_sort_rna_df['symbol'])
        tailed_cpnum_cellline_list = list(tailed_sort_cpnum_df.columns)
        tailed_cpnum_gene_list = list(tailed_sort_cpnum_df['symbol']) 
        error = 0
        for (rna_cl, cpnum_cl) in zip(tailed_rna_cellline_list, tailed_cpnum_cellline_list):
            if rna_cl != cpnum_cl: error = 1
        for (rna_gene, cpnum_gene) in zip(tailed_rna_gene_list, tailed_cpnum_gene_list):
            if rna_gene != cpnum_gene: error = 2
        if error == 0: 
            print('--- CONFIRMED ON IDENTICAL OF [RNA_Seq, CpNum] ---')
        print('--- GDSC(RNA_Seq, CpNum) INTERSECTION [GENES, CELLLINES]: ' + str(tailed_sort_cpnum_df.shape) + ' ---')


    # FIND CELLLINES INTERSECTION BETWEEN [GDSC2_dl_input / CpNum, RNA_Seq], THEN CONDENSE [mid_GDSC2_dl_input]
    # (FINALLY 38227 POINTS INPUT)
    def cellline_intersect_input_condense():
        # CELLLINES IN [mid_GDSC2_dl_input]
        mid_dl_input_df = pd.read_table('./datainfo/mid_data/mid_GDSC2_dl_input.txt', delimiter = ',')
        input_cellline_name = list(mid_dl_input_df['CELL_LINE_NAME'])
        input_cellline_name_list = []
        for cellline in input_cellline_name:
            if cellline not in input_cellline_name_list:
                input_cellline_name_list.append(cellline)
        # CELLLINES IN [RNA_Seq, CpNum]
        rna_df = pd.read_csv('./GDSC/rnaseq_20191101/tailed_rnaseq_fpkm_20191101.csv')
        rna_cellline_list = list(rna_df.columns)
        rna_cellline_list.remove('gene_id')
        rna_cellline_list.remove('symbol')
        cpnum_df = pd.read_csv('./GDSC/cnv_20191101/tailed_cnv_gistic_20191101.csv')
        cpnum_cellline_list = list(cpnum_df.columns)
        cpnum_cellline_list.remove('gene_id')
        cpnum_cellline_list.remove('symbol')
        # GET INTERSECTION OF [GDSC2_dl_input, RNA_Seq]
        rna_cellline_set = set(rna_cellline_list)
        input_cellline_set = set(input_cellline_name_list)
        common_cellline_list = list(rna_cellline_set.intersection(input_cellline_set))
        # REMOVE CELLLINES FOR [CpNum, RNA_Seq]
        print('\n[DL INPUT] REMOVING RNA-Seq/CpNum OUTER CELL LINES ...')
        cellline_deletion_list = []
        for cellline in rna_cellline_list:
            if cellline not in common_cellline_list:
                cellline_deletion_list.append(cellline)
        tail_cellline_rna_df = rna_df.drop(columns = cellline_deletion_list)
        tail_cellline_rna_df.to_csv('./datainfo/mid_data/intersect_rnaseq_fpkm_20191101.csv', index = False, header = True)
        tail_cellline_cpnum_df = cpnum_df.drop(columns = cellline_deletion_list)
        tail_cellline_cpnum_df.to_csv('./datainfo/mid_data/intersect_cnv_gistic_20191101.csv', index = False, header = True)
        print('--- GDSC(RNA_Seq, CpNum, mid_dl_input) INTERSECTION [GENES, CELLLINES]: ' + str(tail_cellline_cpnum_df.shape) + '---')
        # REMOVE CELLLINES FOR [mid_GDSC2_dl_input]
        print('[DL INPUT] REMOVING mid_GDSC2_dl_input OUTER CELL LINES ...')
        mid_dl_input_deletion_list = []
        for row in mid_dl_input_df.itertuples():
            if row[1] not in common_cellline_list:
                mid_dl_input_deletion_list.append(row[0])
        final_dl_input_df = mid_dl_input_df.drop(mid_dl_input_df.index[mid_dl_input_deletion_list]).reset_index(drop = True)
        final_dl_input_df.to_csv('./datainfo/mid_data/final_GDSC2_dl_input.txt', index = False, header = True)
        print('--- DL FINAL INPUT POINTS: ' + str(final_dl_input_df.shape[0]) + ' ---')


    # FIND GENES INTERSECTION BETWEEN [Selected_Kegg_Pathways2 / CpNum, RNA_Seq], COMMON GENES [TT1954 -> TF929]
    def gene_intersect_gdsc_condense():
        gene_pathway_df = pd.read_table('./datainfo/init_data/Selected_Kegg_Pathways2.txt')
        gene_list = list(gene_pathway_df['AllGenes'])
        intersect_rna_df = pd.read_csv('./datainfo/mid_data/intersect_rnaseq_fpkm_20191101.csv')
        intersect_cpnum_df = pd.read_csv('./datainfo/mid_data/intersect_cnv_gistic_20191101.csv')
        rna_gene_list = list(intersect_rna_df['symbol'])
        rna_gene_set = set(rna_gene_list)
        pathway_gene_set = set(gene_list)
        common_gene_list = list(rna_gene_set.intersection(pathway_gene_set))
        print('\n[PATHWAY GENE] REMOVING RNA-Seq/CpNum OUTER GENES ...')
        gene_deletion_list = []
        for gene in rna_gene_list:
            if gene not in common_gene_list:
                gene_deletion_list.append(gene)
        rna_gene_deletion_index = []
        for row in intersect_rna_df.itertuples():
            if row[2] in gene_deletion_list:
                rna_gene_deletion_index.append(row[0])
        tailed_rna_df = intersect_rna_df.drop(rna_gene_deletion_index).reset_index(drop = True)
        tailed_rna_df = tailed_rna_df.fillna(0.0)
        tailed_rna_df.to_csv('./datainfo/filtered_data/tailed_rnaseq_fpkm_20191101.csv', index = False, header = True)
        # print(tailed_rna_df)
        cpnum_gene_deletion_index = []
        for row in intersect_cpnum_df.itertuples():
            if row[2] in gene_deletion_list:
                cpnum_gene_deletion_index.append(row[0])
        tailed_cpnum_df = intersect_cpnum_df.drop(cpnum_gene_deletion_index).reset_index(drop = True)
        tailed_cpnum_df.to_csv('./datainfo/filtered_data/tailed_cnv_gistic_20191101.csv', index = False, header = True)
        # print(tailed_cpnum_df)
        # print(tailed_rna_df.isnull().sum())
        print('--- GDSC(RNA_Seq, CpNum) FINAL [GENES, CELLLINES]: ' + str(tailed_cpnum_df.shape) + ' ---')


    # [GDSC RNA_Seq/CpNum GENES : DRUG_TAR GENES]  KEY : VALUE
    def gene_target_num_dict():
        drug_dict, drug_num_dict, target_dict, target_num_dict = ParseFile.drug_target()
        rna_df = pd.read_csv('./datainfo/filtered_data/tailed_rnaseq_fpkm_20191101.csv')
        print(rna_df.shape)
        # print(target_dict)
        gene_target_num_dict = {}
        count = 0
        num = 0
        for row in rna_df.itertuples():
            if row[2] not in target_dict.keys(): 
                map_index = -1
                count += 1
            else:
                map_index = target_dict[row[2]]
                num += 1
            gene_target_num_dict[row[0]] = map_index
        print(count)
        print(num)
        np.save('./datainfo/filtered_data/gene_target_num_dict.npy', gene_target_num_dict)
        return gene_target_num_dict
    

    # FORM ADAJACENT FILE
    # THEN CONVERT INTO MATRIX (GENE x PATHWAY) (LIST -> SORTED -> DICT -> MATRIX)
    def gene_pathway():
        gene_pathway_df = pd.read_table('./datainfo/init_data/Selected_Kegg_Pathways2.txt')
        gene_pathway_df = gene_pathway_df.sort_values(by = ['AllGenes']).reset_index(drop = True)
        gene_list = list(gene_pathway_df['AllGenes'])
        gene_pathway_df = gene_pathway_df.drop(['AllGenes'], axis = 1)
        pathway_list = list(gene_pathway_df.columns)
        # CONVERT SORTED LIST TO DICT WITH INDEX
        gene_dict = {gene_list[i] : i for i in range(len(gene_list))}
        gene_num_dict = {i : gene_list[i] for i in range(len(gene_list))}
        pathway_dict = {pathway_list[i] : i for i in range(len(pathway_list))}
        pathway_num_dict = {i : pathway_list[i] for i in range(len(pathway_list))}
        # ITERATE THE DATAFRAME TO DEFINE CONNECTIONS BETWEEN GENES AND PATHWAYS
        gene_pathway_matrix = np.zeros((len(gene_list), len(pathway_list))).astype(int)
        for gene_row in gene_pathway_df.itertuples():
            pathway_index = 0
            for gene in gene_row[1:]:
                if gene != 'test':
                    gene_pathway_matrix[gene_dict[gene], pathway_index] = 1.0
                pathway_index += 1
        np.save('./datainfo/mid_data/gene_pathway_matrix.npy', gene_pathway_matrix)
        order_gene_pathway_df = pd.DataFrame(data = gene_pathway_matrix,
            index = [gene for gene in gene_list],
            columns = [pathway for pathway in pathway_list])
        order_gene_pathway_df.to_csv('./datainfo/mid_data/Ordered_Selected_Kegg_Pathways2.csv', index = True, header = True)
        print(order_gene_pathway_df.shape)
        

    # PARSE THOSE GENES NOT IN [RNA_Seq / CpNum]
    def gene_pathway_parse():
        order_gene_pathway_df = pd.read_csv('./datainfo/mid_data/Ordered_Selected_Kegg_Pathways2.csv')
        rna_df = pd.read_csv('./datainfo/filtered_data/tailed_rnaseq_fpkm_20191101.csv')
        cpnum_df = pd.read_csv('./datainfo/filtered_data/tailed_cnv_gistic_20191101.csv')
        rna_gene_list = list(rna_df['symbol'])
        deletion_list = []
        for row in order_gene_pathway_df.itertuples():
            if row[1] not in rna_gene_list:
                deletion_list.append(row[0])
        tailed_gene_pathway_df = order_gene_pathway_df.drop(deletion_list).reset_index(drop = True)
        tailed_gene_pathway_df.to_csv('./datainfo/filtered_data/Tailed_Selected_Kegg_Pathways2.csv', index = False, header = True)
        tailed_gene_pathway_df = tailed_gene_pathway_df.drop(columns = ['Unnamed: 0'])
        print(tailed_gene_pathway_df.shape)
        # CONFIRMATION ON TAILED FILES' [pathway_gene, rna_seq]  GENES IDNETICAL
        tailed_rna_cellline_list = list(rna_df.columns)
        tailed_rna_gene_list = list(rna_df['symbol'])
        tailed_cpnum_cellline_list = list(cpnum_df.columns)
        tailed_cpnum_gene_list = list(cpnum_df['symbol']) 
        error = 0
        for (rna_cl, cpnum_cl) in zip(tailed_rna_cellline_list, tailed_cpnum_cellline_list):
            if rna_cl != cpnum_cl: error = 1
        for (rna_gene, cpnum_gene) in zip(tailed_rna_gene_list, tailed_cpnum_gene_list):
            if rna_gene != cpnum_gene: error = 2
        if error == 0: 
            print('--- CONFIRMED ON IDENTICAL OF [RNA_Seq, CpNum] ---')
        np.save('./datainfo/filtered_data/gene_pathway_matrix.npy', tailed_gene_pathway_df.values)
        

# FORM DRUG MAP BETWEEN [GDSC2_dl_input, drug_tar_drugBank]
def pre_drug_manual():
    ParseFile.second_input_condense()
    ParseFile.drug_map()
    # AFTER GET [/init_data/drug_map.csv] WITH AUTO MAP -> MANUAL MAP

# REMOVE DRUGS IN [GDSC2_dl_input] NOT IN [drug_tar_drugBank]
def pre_drug_parse():
    ParseFile.drug_map_dict()
    ParseFile.drug_target()
    ParseFile.input_drug_condense()

# INPUT CELLLINE CONDENSE BY INTERSECTION BETWEEN [mid_GDSC2_dl_input, CpNum, RNA_Seq]
def pre_cellline_parse():
    rna_filter = True
    cpnum_filter = False
    ParseFile.rna_cpnum_filter(rna_filter, cpnum_filter)
    ParseFile.rna_cpnum_intersect(rna_filter, cpnum_filter)
    ParseFile.cellline_intersect_input_condense()
    
# REMOVE GENES IN [CpNum / RNA_Seq] NOT IN [Selected_Kegg_Pathways2]
def pre_gene_parse():
    ParseFile.gene_intersect_gdsc_condense()

# BUILD DICTIONARY FOR
def pre_relation():
    ParseFile.gene_target_num_dict()
    ParseFile.gene_pathway()
    ParseFile.gene_pathway_parse()

# FINAL INPUT PARSE WITH ALL ZERO ON DRUG TARGET
def zero_final():
    ParseFile.input_drug_gene_condense()

def k_fold_split(random_mode, k, place_num):
    dir_opt = '/datainfo2'
    if random_mode == True:
        ParseFile.input_random_condense()
        ParseFile.random_label()
    ParseFile.split_k_fold(k, place_num)


if __name__ == "__main__":
    # # Manually Fix the Drug Map Problem
    # pre_drug_manual()
    # pre_drug_parse()

    # # Go Over the Data Parse Chart Flow
    # pre_cellline_parse()
    # pre_gene_parse()
    # pre_relation()
    # zero_final()

    # DOING K-FOLD VALIDATION IN 100% DATASET
    random_mode = False
    k = 5
    place_num = 1
    k_fold_split(random_mode, k, place_num)

    # ParseFile.random_label()