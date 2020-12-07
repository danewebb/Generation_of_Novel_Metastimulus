import os
import pickle
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import plot_model
from matplotlib import pyplot as plt

class Atom_FFNN:
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



    def mod_labels(self):

        from sklearn import preprocessing
        # self.train_labels = preprocessing.normalize(self.train_labels)

        # print(self.train_labels)
        # zeros have too great of an impact on zero column
        # for ii in range(len(self.train_labels)):
        #     if self.train_labels[ii,0] == 0 or self.train_labels[ii,0] == -0.0:
        #         self.train_labels[ii,0] = .9

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



    # def padding(self, data):
    #     # array padding with zeros
    #     padded_data = keras.preprocessing.sequence.pad_sequences(
    #         data, padding='post'
    #     )
    #     data_size = padded_data.shape
    #
    #     return padded_data, data_size

    def build_regression_model(self):

        model = keras.Sequential([
            # layers.Dense(self.dense3, activation='relu'),
            # layers.GlobalAveragePooling1D(),

            # layers.Dense(self.dense2, activation='relu', input_shape=(2,)),
            layers.Dense(self.dense2, input_shape=(self.dense2,)),
            layers.Dense(self.hidden, activation='relu'),
            layers.Dropout(self.drop_per),
            layers.Dense(self.dense1, activation='linear')
        ])

        model.compile(

            optimizer='adam',
            loss=tf.keras.losses.mean_squared_error,
            metrics=['mean_squared_error']

        )

        return model

    def build_classification_model(self):

        model = keras.Sequential([
            # layers.Dense(self.dense3, activation='relu'),
            # layers.GlobalAveragePooling1D(),

            # layers.Dense(self.dense2, activation='relu', input_shape=(2,)),
            layers.Dense(self.dense2, input_shape=(self.dense2,)),
            layers.Dense(self.hidden, activation='relu'),
            layers.Dropout(self.drop_per),
            layers.Dense(self.dense1, activation='sigmoid')
        ])

        model.compile(

            optimizer='adam',
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy']

        )
        return model



    def train(self):
        try:
            self.history = self.model.fit(
                self.data, self.train_labels, epochs=self.epochs, batch_size=self.batch_size, shuffle=True # shuffles batches
                ,verbose=1 # supresses progress bar
            )
        except ValueError:
            self.data = self.data.T
            self.history = self.model.fit(
                self.data, self.train_labels, epochs=self.epochs, batch_size=self.batch_size, shuffle=True
                # shuffles batches
                , verbose=1  # supresses progress bar
            )
        return self.history



    def test(self):
        try:
            result = self.model.evaluate(self.test_data, self.test_labels, batch_size=self.batch_size)

        except:
            self.test_data = self.test_data.T # transpose data
            result = self.model.evaluate(self.test_data, self.test_labels, batch_size=self.batch_size)

        if len(result) > 2:
            print(f'Test results: Loss= {result[0]:.3f}, Accuracy = {100*result[1]:.3f}%')
            return result[0], result[1]
        else:
            print(f'Test results: Loss= {result[0]:.3f}')
            return result

    def predict(self):
        # self.data = self.data.T
        try:
            result = self.model.predict(self.data, batch_size=self.batch_size)
        except:
            self.data = self.data.T
            result = self.model.predict(self.data, batch_size=self.batch_size)
        return result






    def save_model(self):

        keras.models.save_model(
            self.model, self.save_model_path
        )


    def training_results(self, param='', graph_dict_path='', save_file=''):
        param_dict = dict()

        param_dict['loss'] = self.history.history['loss']
        param_dict['epochs'] = self.epochs

        try:
            param_dict['acc'] = self.history.history['accuracy']
        except:
            pass

        if graph_dict_path == '' or not os.path.exists(graph_dict_path):
            graph_dict = dict()
            graph_dict[param] = param_dict


        else:
            with open(graph_dict_path, 'rb') as f2:
                graph_dict = pickle.load(f2)
            graph_dict[param] = param_dict



        with open(save_file, 'wb') as f3:
            pickle.dump(graph_dict, f3)


    def test_classification_results(self, graph_dict, param):
        loss = []
        acc = []
        param_dict = dict()

        for ep in range(self.epochs):
            l, a = AFF.test()
            loss.append(l)
            acc.append(a)

        param_dict['loss'] = loss
        param_dict['acc'] = acc
        param_dict['epochs'] = self.epochs
        graph_dict[param] = param_dict

        return graph_dict

    def test_regression_results(self, regress_graph_dict, param):
        loss = []
        param_dict = dict()

        for ep in range(self.epochs):
            l, a = AFF.test()
            loss.append(l)


        param_dict['loss'] = loss
        param_dict['epochs'] = self.epochs
        regress_graph_dict[param] = param_dict

        return regress_graph_dict





if __name__ == '__main__':
    with tf.device('/cpu:0'):
        set_batch_size = 20
        set_epochs = 100
        AFF = Atom_FFNN(
            data_path=r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\atom_vectors_avg.pkl', # local atom vector path
            train_label_path=r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\train_labels_proj.pkl',
            batch_size=set_batch_size,
            epochs=set_epochs,
            regression=True,
            # classification=True,
            model_path=r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_LV_NN\model100_proj',
            # save_model_path=r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_LV_NN\model100_proj',
            test_label_path=r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\null_train_labels_proj.pkl',
            test_path=r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\atom_vectors_avg.pkl',
            dense_out=2,
            hidden=100,
            dense_in=2

        )



        # training and saving
        # AFF.mod_labels() # only for regression
        # history = AFF.train()
        # print(
        #     'hello world'
        # )
        # AFF.save_model()

        # param = 'ff_100ep_train_ricoword_BOWavg_class'
        # AFF.training_results(param,
        #                     graph_dict_path= r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_LV_NN\graphing_data.pkl',
        #                     save_file=r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_LV_NN\graphing_data.pkl')



        # testing and saving probabilistic

        # param = 'ff_100ep_nullset_ricoword_BOWavg_class'
        # with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_LV_NN\graphing_data.pkl', 'rb') as f1:
        #     graph_dict = pickle.load(f1)
        # AFF.test_classification_results(graph_dict, param)
        #
        #
        # with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_LV_NN\graphing_data.pkl', 'wb') as f2:
        #     pickle.dump(graph_dict, f2)


        # testing and saving regression


        # param = 'ff_100ep_train_ricoword_BOWavg_proj'
        # AFF.training_results(param,
        #                     graph_dict_path= r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_LV_NN\regress_graph_data.pkl',
        #                     save_file=r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_LV_NN\regress_graph_data.pkl')

        param = 'ff_100ep_nullset_ricoword_BOWavg_proj'
        with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_LV_NN\regress_graph_data.pkl', 'rb') as f3:
            regress_graph_dict = pickle.load(f3)

        regress_graph_dict = AFF.test_regression_results(regress_graph_dict, param)

        with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_LV_NN\regress_graph_data.pkl', 'wb') as f4:
            pickle.dump(regress_graph_dict, f4)


        # use case
        # results = AFF.predict()
        #
        # print(results)