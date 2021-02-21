import numpy as np
import pandas as pd

class GenMatrix():
    def __init__(self):
        pass

    # AUTO GENERATE FEATURE_GENE CONNECTION IN PRE_INPUT LAYER
    def feature_gene_matrix(num_feature, num_gene):
        matrixA = np.zeros((num_feature * num_gene, num_gene))
        count = 0
        for i in range(num_gene):
            matrixA[count:count+num_feature, i] = 1
            count += num_feature
        return matrixA

    # FORM MATRIXB ACCORDING TO GENE_PATHWAY CONNECTIONS FROM CELLLINE_GENE
    def gene_pathway_matrix():
        matrixB = np.load('./datainfo/filtered_data/gene_pathway_matrix.npy')
        matrixB = matrixB.astype(float)
        return matrixB