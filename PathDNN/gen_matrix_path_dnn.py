import numpy as np
import pandas as pd

class GenMatrix():
    def __init__(self):
        pass

    # FORM MATRIXB ACCORDING TO GENE_PATHWAY CONNECTIONS FROM CELLLINE_GENE
    def gene_pathway_matrix():
        matrixB = np.load('./datainfo/filtered_data/gene_pathway_matrix.npy')
        matrixB = matrixB.astype(float)
        matrixB = np.vstack((matrixB, matrixB))
        return matrixB