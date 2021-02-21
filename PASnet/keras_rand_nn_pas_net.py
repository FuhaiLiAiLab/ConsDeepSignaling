import os
import shutil
import pickle
import simplejson
import numpy as np
import pandas as pd
import keras.backend as K
import matplotlib.pyplot as plt

from keras import Model
from keras.optimizers import Adam
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Input, Dropout, LeakyReLU

import innvestigate
import tensorflow as tf

from gen_matrix import GenMatrix
from parse_file_pas_net import ParseFile
from load_data_pas_net import LoadData


# BUILD A NOT FULLY CONNECTED NN
class CustomConnected(Dense):
    def __init__(self, units, connections, **kwargs):
        # THIS IS MATRIX_A
        self.connections = connections      
        # INITALIZE THE ORIGINAL DENSE WITH ALL THE USUAL ARGUMENTS   
        super(CustomConnected, self).__init__(units, **kwargs)  

    def call(self, inputs):
        output = K.dot(inputs, self.kernel * self.connections)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


class RandNN():
    def __init__(self):
        pass

    # DECOMPOSE NEURAL NETWORK
    def keras_rand_nn(self, matrixB, num_gene, num_pathway, layer1):
        num_gene, num_pathway = matrixB.shape

        input_x = Input(shape = (num_gene, ))
        pathway = CustomConnected(num_pathway, matrixB)(input_x)
        input_model = Model(input_x, pathway)

        pathway_info = input_model(input_x)
        pathway_input = Input(shape = (num_pathway, ))
        # ADD DENSE LAYERS
        output1 = Dense(layer1, activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.01))(pathway_input)
        output1 = Dropout(0.1)(output1)
        output2 = Dense(1, activation='linear',
            kernel_regularizer=tf.keras.regularizers.l2(0.01))(output1)
        pathway_model= Model(pathway_input, output2)

        output2 = pathway_model(pathway_info)
        model = Model(input_x, output2)

        return input_model, pathway_model, model


class RunRandNN():
    def __init__(self, model, dir_opt):
        self.model = model
        self.dir_opt = dir_opt

    # TRAIN DECOMPOSED DEEP NEURAL NETWORK
    def train(self, input_num, epoch, batch_size, verbose, learning_rate, end_epoch):
        model = self.model
        dir_opt = self.dir_opt
        print("--------------LEARNING RATE: " + str(learning_rate) + "--------------")
        model.compile(loss='mean_squared_error',
                    optimizer = Adam(lr = learning_rate),
                    metrics=['mse', 'accuracy']) 
        # CLEAN RESULT PREVIOUS EPOCH_I_PRED FILES
        folder_name = 'epoch_' + str(end_epoch)
        path = '.' + dir_opt + '/result/%s' % (folder_name)
        unit = 1
        if os.path.exists('.' + dir_opt + '/result') == False:
            os.mkdir('.' + dir_opt + '/result')
        while os.path.exists(path):
            path = '.' + dir_opt + '/result/%s_%d' % (folder_name, unit)
            unit += 1
        os.mkdir(path)
        # TRAIN MODEL IN EPOCH ITERATIONS
        epoch_mse_list = []
        epoch_pearson_list = []
        for i in range(epoch):
            print('--------------EPOCH: ' + str(i) + ' --------------') 
            epoch_train_pred = np.zeros((1, 1))
            upper_index = 0
            batch_mse_list = []
            for index in range(0, input_num, batch_size):
                if (index + batch_size) < input_num:
                    upper_index = index + batch_size
                else:
                    upper_index = input_num
                xTr_batch, yTr_batch = LoadData(dir_opt).load_train(index, upper_index)
                history = model.fit(xTr_batch, yTr_batch, epochs = 1, verbose = verbose)
                # PRESERVE MSE FOR EVERY BATCH
                # print(history.history)
                print('EPOCH MSE LOSS: ' + str(history.history['mean_squared_error'][0]))
                batch_mse_list.append(history.history['mean_squared_error'])
                # PRESERVE PREDICTION OF TRAINING MODEL IN EVERY BATCH
                train_batch_pred = np.array(model.predict(xTr_batch))
                epoch_train_pred = np.vstack((epoch_train_pred, train_batch_pred))
            # PRESERVE MSE FOR EVERY EPOCH
            print(np.mean(batch_mse_list)) # mse shuold use weight average, here mostly same, just ignore
            epoch_mse_list.append(np.mean(batch_mse_list))
            # SAVE RESULT FOR EVERY EPOCH PREDICTION
            epoch_train_pred = np.delete(epoch_train_pred, 0, axis = 0)
            np.save(path + '/epoch_' + str(i) + '_pred.npy', epoch_train_pred)
            # PRESERVE PEARSON CORR FOR EVERY EPOCH
            tmp_training_input_df = pd.read_csv('.' + dir_opt + '/filtered_data/TrainingInput.txt', delimiter = ',')
            final_row, final_col = tmp_training_input_df.shape
            epoch_train_pred_lists = list(epoch_train_pred)
            epoch_train_pred_list = [item for elem in epoch_train_pred_lists for item in elem]
            tmp_training_input_df.insert(final_col, 'Pred Score', epoch_train_pred_list, True)
            epoch_pearson = tmp_training_input_df.corr(method = 'pearson')
            epoch_pearson_list.append(epoch_pearson)
            print(epoch_pearson)
        # TRAINNING OUTPUT PRED
        final_train_input_df = pd.read_csv('.' + dir_opt + '/filtered_data/TrainingInput.txt', delimiter = ',')
        final_row, final_col = final_train_input_df.shape
        epoch_train_pred_lists = list(epoch_train_pred)
        epoch_train_pred_list = [item for elem in epoch_train_pred_lists for item in elem]
        final_train_input_df.insert(final_col, 'Pred Score', epoch_train_pred_list, True)
        final_train_input_df.to_csv(path + '/PredTrainingInput.txt', index = False, header = True)
        print(epoch_mse_list)
        print(epoch_pearson_list)
        # PRESERVE TRAINING MODEL RESULTS [MSE, PEARSON]
        fp = open(path + '/epoch_result_list.txt', 'w')
        simplejson.dump(str(epoch_mse_list), fp)
        simplejson.dump(str(epoch_pearson_list), fp)
        fp.close()
        # PRESERVE TRAINED DECOMPOSED MODEL
        model.save_weights(path + '/model.h5')
        # PRESERVE TRAINED MODEL EACH LAYER WEIGHT PARAMETERS
        layer_list = []
        num_layer = len(model.layers)
        for i in range(num_layer):
            layer_list.append(model.get_layer(index = i).get_weights())
        with open(path + '/layer_list.txt', 'wb') as filelayer:
            pickle.dump(layer_list, filelayer)
        return model, history, path

    def test(self, verbose, path):
        model = self.model
        dir_opt = self.dir_opt
        # xTe, yTe = LoadData(dir_opt).load_test()
        xTe = np.load('./datainfo/post_data/xTe.npy')
        yTe = np.load('./datainfo/post_data/yTe.npy')
        # TEST OUTPUT PRED 
        y_pred = model.predict(xTe)
        y_pred_list = [item for elem in y_pred for item in elem]
        score = model.evaluate(xTe, yTe, verbose = verbose)
        final_test_input_df = pd.read_csv('.' + dir_opt + '/filtered_data/TestInput.txt', delimiter = ',')
        final_row, final_col = final_test_input_df.shape
        final_test_input_df.insert(final_col, 'Pred Score', y_pred_list, True)
        final_test_input_df.to_csv(path + '/PredTestInput.txt', index = False, header = True)
        # ANALYSE PEARSON CORR
        test_pearson = final_test_input_df.corr(method = 'pearson')
        print(score)
        print(test_pearson)
        return y_pred, score