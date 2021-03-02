import os
import pickle
import innvestigate
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import Model
from keras.optimizers import Adam
from keras.models import load_model
from gen_matrix import GenMatrix
from parse_file_flabel import ParseFile
from load_data_flabel import LoadData
from analysis import Analyse
from keras_rand_nn_flabel import RandNN, RunRandNN

# BUILD DECOMPOSED MODEL
def build_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3):
    input_model, gene_model, pathway_model, model = RandNN().keras_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3)
    return input_model, gene_model, pathway_model, model

# RUN MODEL (AUTO UPDATE WEIGHT)
def run_rand_nn(model, dir_opt, matrixA, matrixB, input_num, epoch, batch_size, verbose, learning_rate, end_epoch):
    model, history, path = RunRandNN(model, dir_opt).train(input_num, epoch, batch_size, verbose, learning_rate, end_epoch)
    return model, history, path

# GET MODEL IMMEDIATELY FROM TRAINED MODEL
def auto_test_rand_nn(model, dir_opt, verbose, path):
    y_pred, score = RunRandNN(model, dir_opt).test(verbose, path)

def manual_test_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path, dir_opt, learning_rate):
    # MANUAL REBUILD THE MODEL
    input_model, gene_model, pathway_model, model = build_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3)
    with open(path + '/layer_list.txt', 'rb') as filelayer:
        layer_list = pickle.load(filelayer)
    model.compile(loss='mean_squared_error',
                    optimizer=Adam(lr=learning_rate),
                    metrics=['mse', 'accuracy']) 
    xTmp, yTmp = LoadData(dir_opt).load_train(0, 1)
    model.fit(xTmp, yTmp, epochs = 1, validation_split = 1, verbose = 0)
    num_layer = len(model.layers)
    for i in range(num_layer):
        model.get_layer(index = i).set_weights(layer_list[i])
    # PREDICT MODEL USING [xTe, yTe]
    verbose = 1
    y_pred, score = RunRandNN(model, dir_opt).test(verbose, path)

# CONTINUE RUN MODEL
def continue_run_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path,
                dir_opt, input_num, epoch, batch_size, verbose, learning_rate, end_epoch):
    # REBUILD DECOMPOSED MODEL FROM SAVED MODEL
    input_model, gene_model, pathway_model, model = build_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3)
    with open(path + '/layer_list.txt', 'rb') as filelayer:
        layer_list = pickle.load(filelayer)
    model.compile(loss='mean_squared_error',
                    optimizer=Adam(lr=learning_rate),
                    metrics=['mse', 'accuracy']) 
    xTmp, yTmp = LoadData(dir_opt).load_train(0, 1)
    model.fit(xTmp, yTmp, epochs = 1, validation_split = 1, verbose = 0)
    num_layer = len(model.layers)
    for i in range(num_layer):
        model.get_layer(index = i).set_weights(layer_list[i])
    # RUN MODEL (AUTO UPDATE WEIGHT)    
    model, history, path = run_rand_nn(model, dir_opt, matrixA, matrixB, input_num, epoch, batch_size, verbose, learning_rate, end_epoch)
    return model, history, path


#################################################################################################

def initial_run_dnn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3,
                dir_opt, input_num, epoch, batch_size, verbose, learning_rate): 
    # RUN DEEP NERUAL NETWORKs
    print('RUNING DEEP NERUAL NETWORK...')
    end_epoch = epoch
    input_model, gene_model, pathway_model, model = build_rand_nn(matrixA, matrixB,
                num_gene, num_pathway, layer1, layer2, layer3)
    model, history, path = run_rand_nn(model, dir_opt, matrixA, matrixB, input_num, epoch, batch_size, verbose, learning_rate, end_epoch)

def schedule_cont_run_dnn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path,
                dir_opt, input_num, epoch, batch_size, verbose, learning_rate, start_epoch):
    # CONTINUE RUN TRAINED DEEP NEURAL NETWORK
    print('CONTINUE RUNING DEEP NERUAL NETWORK...')
    end_epoch = start_epoch + epoch
    path = '.' + dir_opt + '/result/epoch_%d' % (start_epoch)
    model, history, path = continue_run_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path,
                dir_opt, input_num, epoch, batch_size, verbose, learning_rate, end_epoch)
    auto_test_rand_nn(model, dir_opt, verbose, path)


def learn_schedule_dnn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3,
                dir_opt, input_num, batch_size, verbose):
    schedule_lr_list = [0.001, 0.0001, 0.00005, 0.00001, 0.000001]
    schedule_epoch_list = [29, 10, 10, 20, 30]
    schedule_start_epoch_list = [29, 39, 49, 69] # FILE ENDS WITH [29, 39, 49, 89, 99]
    # INITIAL SCHEDULING
    epoch = schedule_epoch_list[0]
    learning_rate = schedule_lr_list[0]
    initial_run_dnn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3,
                dir_opt, input_num, epoch, batch_size, verbose, learning_rate)
    # CONTINUE SCHEDULING
    count = 1
    for start_epoch in schedule_start_epoch_list:
        epoch = schedule_epoch_list[count]
        learning_rate = schedule_lr_list[count]
        count = count + 1
        path = '.' + dir_opt + '/result/epoch_%d' % (start_epoch)
        schedule_cont_run_dnn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path,
                    dir_opt, input_num, epoch, batch_size, verbose, learning_rate, start_epoch)


if __name__ == "__main__":
    # READ [NUM_FEATIRES/NUM_GENES/ NUM_PATHWAY] FROM DEEP_LEARNING_INPUT, RNA_SEQ, GENE_PATHWAY
    print('READING DIMS...')
    dir_opt = '/datainfo'
    train_input_df, input_num, num_feature, rna_df, cpnum_df, num_gene, num_pathway = LoadData(dir_opt).pre_load_train()

    # BUILD NEURAL NETWORK
    print('BUILDING CUSTOMED NERUAL NETWORK...')
    layer0 = num_pathway
    layer1 = 256
    layer2 = 128
    layer3 = 32
    matrixA = GenMatrix.feature_gene_matrix(num_feature, num_gene)
    matrixB = GenMatrix.gene_pathway_matrix()
    print('-----MATRIX A - FEATURES OF GENE SHAPE-----')
    print(matrixA.shape)
    print('-----MATRIX B - GENE PATHWAY SHAPE-----')
    print(matrixB)

    # # RUN DEEP NERUAL NETWORKs
    # print('RUNING DEEP NERUAL NETWORK...')
    # epoch = 5
    # batch_size = 256
    # verbose = 0
    # learning_rate = 0.001
    # end_epoch = 5
    # input_model, gene_model, pathway_model, model = build_rand_nn(matrixA, matrixB,
    #             num_gene, num_pathway, layer1, layer2, layer3)
    # model, history, path = run_rand_nn(model, dir_opt, matrixA, matrixB, input_num, epoch, batch_size, verbose, learning_rate, end_epoch)

    # # TEST DEEP NERUAL NETWORK MODEL
    # # AUTO TEST NETWORK
    # print('TESTING DEEP NERUAL NETWORK...')
    # auto_test_rand_nn(model, dir_opt, verbose, path)


    # # MANUAL TEST NETWORK
    # path = '.' + dir_opt + '/result/epoch_5'
    # learning_rate = 0.001
    # manual_test_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path, dir_opt, learning_rate)
    

    # # CONTINUE RUN TRAINED DEEP NEURAL NETWORK
    # print('CONTINUE RUNING DEEP NERUAL NETWORK...')
    # path = '.' + dir_opt + '/result/epoch_99_No'
    # epoch = 50
    # batch_size = 256
    # verbose = 0
    # learning_rate = 0.000001
    # end_epoch = 149
    # model, history, path = continue_run_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path,
    #             dir_opt, input_num, epoch, batch_size, verbose, learning_rate, end_epoch)
    # auto_test_rand_nn(model, dir_opt, verbose, path)

    # # MANUAL TEST NETWORK
    # path = '.' + dir_opt + '/result/epoch_30'
    # manual_test_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path, dir_opt)
    



    ############################################################################

    # LEARNING RATE SCHEDULE IN DEEP NEURAL NETWORK
    # batch_size = 256
    # verbose = 0
    # learn_schedule_dnn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3,
    #             dir_opt, input_num, batch_size, verbose)
    
    # MANUAL TEST NETWORK
    path = '.' + dir_opt + '/result/epoch_29'
    learning_rate = 0.001
    manual_test_rand_nn(matrixA, matrixB, num_gene, num_pathway, layer1, layer2, layer3, path, dir_opt, learning_rate)