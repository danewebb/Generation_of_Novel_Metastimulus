import os
import pickle
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import plot_model
from matplotlib import pyplot as plt

class Atom_RNN():

    def __init__(self, data_path='', train_label_path='', test_path='',  test_label_path='',
                 model_path = '', save_model_path='', batch_size=10, epochs=100,
                  drop_per=0.1, dense_out=2, dense_in=2, hidden=50,
                 regression=False, classification=False):
        try:
            with open(data_path, 'rb') as data_file:
                self.data = pickle.load(data_file)
        except ValueError:
            print('Data path does not exist')

        if not isinstance(self.data, np.ndarray):
            self.data = self.data_to_numpy(self.data)
        else:
            TypeError('data should be either ndarray or list')

        self.batch_size = batch_size
        self.epochs = epochs
        self.drop_per = drop_per
        self.dense1 = dense_out
        self.dense2 = dense_in
        self.hidden = hidden




        if test_path != '':

            with open(test_path, 'rb') as test_file:
                self.test_data = pickle.load(test_file)
            if not isinstance(self.test_data, np.ndarray):
                self.test_data = self.data_to_numpy(self.test_data)
            else:
                TypeError('test_data should be either ndarray or list')

            with open(test_label_path, 'rb') as test_label_file:
                self.test_labels = pickle.load(test_label_file)
            if not isinstance(self.test_labels, np.ndarray):
                self.test_labels = self.data_to_numpy(self.test_labels)
            else:
                TypeError('test_labels should be either ndarray or list')


        if train_label_path != '':
            with open(train_label_path, 'rb') as train_label_file:
                self.train_labels = pickle.load(train_label_file)
            if not isinstance(self.train_labels, (np.ndarray)):
                self.train_labels = self.data_to_numpy(self.train_labels)
            else:
                TypeError('train_labels should be either ndarray or list')
        else:
            raise Exception('Train label path is not valid.')



        self.history = dict()


        if save_model_path != '':
            self.save_model_path = save_model_path
        else:
            Warning('save_model_path does not exist. Model will not save.')

        if model_path != '':
            self.model = keras.models.load_model(model_path)
        else:
            print('Building new model')
            if regression:
                self.model = self.build_regression_model()
            elif classification:
                self.model = self.build_classification_model()
            else:
                ValueError('Model is not defined.')


    def data_to_numpy(self, lst):
        # make encoded data into numpy arrays
        holder = []
        for ele in lst:
            # converts list of lists into a numpy array
            if ele == []:
                # check if empty list, not sure why empty lists are in the data.
                ele = [0., 0.]
            temp = np.array(ele)
            temp = temp.reshape((temp.shape[0], 1))
            holder.append(temp)

        arr = np.concatenate(holder, axis=1)


        return arr


    def build_regression_model(self):
        model = keras.Sequential([
            layers.SimpleRNN(
                units=self.units,
                dropout=self.drop_per,
                input_shape=self.input_shape),
            layers.Dense(self.output_size,
                         activation='softmax')
        ])