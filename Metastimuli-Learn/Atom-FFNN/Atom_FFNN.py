import os
import pickle
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import plot_model
from matplotlib import pyplot as plt
from statistics import mean


class Atom_FFNN:
    def __init__(self, data_path='', train_label_path='', test_path='',  test_label_path='',
                 model_path = '', save_model_path='', batch_size=10, epochs=100, nullset_path='', nullset_labels_path='', nullset_labels=None,
                  drop_per=0.1, dense_out=2, dense_in=2, hidden=50, current_dim=0, learning_rate=0.001, seed=24,
                 regression=False, classification=False, normalize=False):


        try:
            with open(data_path, 'rb') as data_file:
                self.data = pickle.load(data_file)
        except ValueError:
            print('Data path does not exist')

        if not isinstance(self.data, np.ndarray):
            self.data = self.data_to_numpy(self.data, dense_out)
        else:
            TypeError('data should be either ndarray or list')

        self.learn_rate = learning_rate
        self.current_dim = current_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.drop_per = drop_per
        self.dense1 = dense_out
        self.dense2 = dense_in
        self.hidden = hidden
        self.seed = seed
        if nullset_path != '':
            # nullset is either train set, test set, or validation set
            with open(nullset_path, 'rb') as null_file:
                self.nullset = pickle.load(null_file)
            if not isinstance(self.nullset, np.ndarray):
                self.nullset = self.data_to_numpy(self.nullset, dense_out)

            # nullset is the randomized labels of the previous dataset
            try:
                with open(nullset_labels_path, 'rb') as null_labels_file:
                    self.nullset_labels = pickle.load(null_labels_file)
                if not isinstance(self.nullset_labels, np.ndarray):
                    self.nullset_labels = self.data_to_numpy(self.nullset_labels, dense_out)
            except:
                self.nullset_labels = nullset_labels






        if test_path != '':

            with open(test_path, 'rb') as test_file:
                self.test_data = pickle.load(test_file)
            if not isinstance(self.test_data, np.ndarray):
                self.test_data = self.data_to_numpy(self.test_data, dense_out)


            with open(test_label_path, 'rb') as test_label_file:
                self.test_labels = pickle.load(test_label_file)
            if not isinstance(self.test_labels, np.ndarray):
                self.test_labels = self.data_to_numpy(self.test_labels, dense_out)





        if train_label_path != '':
            with open(train_label_path, 'rb') as train_label_file:
                self.train_labels = pickle.load(train_label_file)
            if not isinstance(self.train_labels, (np.ndarray)):
                self.train_labels = self.data_to_numpy(self.train_labels, dense_out)





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



        if normalize:
            self.train_labels, self.test_labels, self.nullset_labels = self.norm_all(
                self.train_labels, self.test_labels, self.nullset_labels)


        self.nullset_labels = self.splice_labels(self.nullset_labels, self.current_dim)
        self.train_labels = self.splice_labels(self.train_labels, self.current_dim)
        self.test_labels = self.splice_labels(self.test_labels, self.current_dim)





    def data_to_numpy(self, lst, dims):
        # make encoded data into numpy arrays
        holder = []
        for ele in lst:
            # converts list of lists into a numpy array
            if ele == []:
                # check if empty list, not sure why empty lists are in the data.
                ele = list(np.zeros((dims, 1)))
            temp = np.array(ele)
            temp = temp.reshape((temp.shape[0], 1))
            holder.append(temp)

        arr = np.concatenate(holder, axis=1)


        return arr

    def splice_labels(self, labels, dim):
        try:
            labs = labels[:, dim]
        except:
            labs = None
        return labs

    # def padding(self, data):
    #     # array padding with zeros
    #     padded_data = keras.preprocessing.sequence.pad_sequences(
    #         data, padding='post'
    #     )
    #     data_size = padded_data.shape
    #
    #     return padded_data, data_size

    def build_regression_model(self):
        winit = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=self.seed)
        binit = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=self.seed)
        model = keras.Sequential([
            # layers.Dense(self.dense3, activation='relu'),
            # layers.GlobalAveragePooling1D(),

            # layers.Dense(self.dense2, activation='relu', input_shape=(2,)),
            layers.Dense(self.dense2, input_shape=(self.dense2,), kernel_initializer=winit, bias_initializer=binit),
            # layers.Dense(30, activation='tanh'),
            # layers.Dropout(self.drop_per),
            layers.Dense(260, activation='tanh', kernel_initializer=winit, bias_initializer=binit),
            # layers.Dropout(self.drop_per),
            layers.Dense(13, activation='tanh', kernel_initializer=winit, bias_initializer=binit),
            layers.Dense(self.dense1, activation='linear', kernel_initializer=winit, bias_initializer=binit)
        ])

        sgd = keras.optimizers.SGD(learning_rate=self.learn_rate)

        model.compile(

            optimizer=sgd,

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

    # def train_val(self):
    #     try:
    #         self.history = self.model.fit(
    #             self.data, self.train_labels, epochs=self.epochs, batch_size=self.batch_size, shuffle=True # shuffles batches
    #             ,verbose=1 # supresses progress bar
    #             , validation_data=(self.val_data, self.val_labels)
    #             , validation_steps=int(len(self.val_data) / self.batch_size)
    #         )
    #     except ValueError:
    #         self.val_data = self.val_data.T
    #         self.data = self.data.T
    #         self.history = self.model.fit(
    #             self.data, self.train_labels, epochs=self.epochs, batch_size=self.batch_size, shuffle=True
    #             # shuffles batches
    #             , verbose=1  # supresses progress bar
    #             , validation_data=(self.val_data, self.val_labels)
    #             , validation_steps=int(len(self.val_data)/self.batch_size)
    #         )
    #     return self.history

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

    def test_nullset(self):
        try:
            result = self.model.evaluate(self.nullset, self.nullset_labels, batch_size=self.batch_size)

        except:
            self.nullset = self.nullset.T # transpose data
            result = self.model.evaluate(self.nullset, self.nullset_labels, batch_size=self.batch_size)

        if len(result) > 2:
            print(f'Nulltest results: Loss= {result[0]:.3f}, Accuracy = {100*result[1]:.3f}%')
            return result[0], result[1]
        else:
            print(f'Nulltest results: Loss= {result[0]:.3f}')
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
        set_batch_size = 5
        set_epochs = 1
        learn_rate = 0.005
        # curdim = 0
        model_paths = [
            r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_30dims\BOWsum_rico\W_100_output_3Dims\model500_3dims_260tanh_00',
            r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_30dims\BOWsum_rico\W_100_output_3Dims\model500_3dims_260tanh_01',
            r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_30dims\BOWsum_rico\W_100_output_3Dims\model500_3dims_260tanh_02',
        #     r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_10dims\BOWavg_rico\model50_10dims_03',
        #     r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_10dims\BOWavg_rico\model50_10dims_04',
        #     r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_10dims\BOWavg_rico\model50_10dims_05',
        #     r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_10dims\BOWavg_rico\model50_10dims_06',
        #     r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_10dims\BOWavg_rico\model50_10dims_07',
        #     r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_10dims\BOWavg_rico\model50_10dims_08',
        #     r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_10dims\BOWavg_rico\model50_10dims_09',
        ]


        tr_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_30dims\BOWsum_rico\W_100_output_3Dims\train.pkl'
        trlabels_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_30dims\BOWsum_rico\W_100_output_3Dims\train_labels.pkl'

        te_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_30dims\BOWsum_rico\W_100_output_3Dims\test.pkl'
        telabels_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_30dims\BOWsum_rico\W_100_output_3Dims\test_labels.pkl'



        output_dimension = len(model_paths)
        for dim in range(output_dimension):
            AFF = Atom_FFNN(
                data_path= tr_path, # local atom vector path
                train_label_path= trlabels_path,
                batch_size=set_batch_size,
                epochs=set_epochs,
                regression=True,
                # classification=True,
                # model_path=model_paths[dim],
                save_model_path=model_paths[dim],
                current_dim=dim,
                # current_dim=curdim,
                test_path=te_path,
                test_label_path=telabels_path,
                nullset_path=te_path,
                # nullset_labels_path=r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\BOWavg_rico\nulltest_set.pkl',
                learning_rate=learn_rate,
                dense_out=1,
                hidden=300,
                dense_in=30,
                drop_per=.1,
                normalize=False
            )

            AFF.save_model()

        # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_10dims\ndelta_rico\W_500_output_3Dims\nulltest_3dims.pkl', 'rb') as f12:
        #     null_arr = pickle.load(f12)

        epochs = 500
        restr = np.empty((output_dimension, epochs))
        reste = np.empty((output_dimension, epochs))
        # null = np.empty((output_dimension, epochs, null_arr.shape[2]))

        null_holder = []
        dict1 = dict()
        dict2 = dict()
        dict3 = dict()
        dict4 = dict()

        for dim in range(output_dimension):
            AFF = Atom_FFNN(
                data_path= tr_path, # local atom vector path
                train_label_path= trlabels_path,
                batch_size=set_batch_size,
                epochs=set_epochs,
                regression=True,
                # classification=True,
                # model_path=model_paths[dim],
                save_model_path=model_paths[dim],
                current_dim=dim,
                # current_dim=curdim,
                test_path=te_path,
                test_label_path=telabels_path,
                nullset_path=te_path,
                # nullset_labels_path=r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\BOWavg_rico\nulltest_set.pkl',
                learning_rate=learn_rate,
                dense_out=1,
                hidden=300,
                dense_in=30,
                drop_per=.1,
                normalize=False
            )
            for ii in range(epochs):
                print(f'Epoch: {ii}')
                history = AFF.train()
                trloss = history.history['loss']
                restr[dim, ii] = trloss[0]

                AFF.save_model()
                keras.backend.clear_session()
                reste[dim, ii], _ = AFF.test()

        #         num_nulls = null_arr.shape[2]
        #         for jj in range(num_nulls):
        #             AFF = Atom_FFNN(
        #                 data_path=tr_path,
        #                 # local atom vector path
        #                 train_label_path=trlabels_path,
        #                 batch_size=set_batch_size,
        #                 epochs=set_epochs,
        #                 regression=True,
        #                 # classification=True,
        #                 model_path=model_paths[dim],
        #                 save_model_path=model_paths[dim],
        #                 # current_dim=dim,
        #                 current_dim=curdim,
        #                 test_path=te_path,
        #                 test_label_path=telabels_path,
        #                 nullset_path=te_path,
        #                 nullset_labels=null_arr[:, :, jj],
        #                 learning_rate=learn_rate,
        #                 dense_out=1,
        #                 hidden=250,
        #                 dense_in=10,
        #                 drop_per=.1,
        #                 normalize=False
        #             )
        #             null[dim, ii, jj], _ = AFF.test_nullset()
        #
        #
        with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_30dims\results_3dims.pkl', 'rb') as f10:
            graph_dict = pickle.load(f10)
        # graph_dict = dict()
        #
        # null_avg = np.average(null, axis=2)


        param_tr = 'ff_500ep_train_rico_BOWsum_w_100_3dim_260tanh-13tanh'
        param_te = 'ff_500ep_test_rico_BOWsum_w_100_3dim_260tanh-13tanh'
        # param_null = 'ff_50ep_nullset_rico10dims_ndelta_w_500_3dim_tanh_02'
        param_pred = 'ff_500ep_pred_rico_BOWsum_w_100_3dim_260tanh-13tanh'


        dict1['loss'] = restr
        dict1['epochs'] = epochs
        graph_dict[param_tr] = dict1

        dict2['loss'] = reste
        dict2['epochs'] = epochs
        graph_dict[param_te] = dict2

        # dict3['loss'] = null_avg
        # dict3['epochs'] = epochs
        # graph_dict[param_null] = dict3


        resp = []
        for dim in range(output_dimension):

            AFF = Atom_FFNN(
                data_path=tr_path,
                # local atom vector path
                train_label_path=trlabels_path,
                batch_size=set_batch_size,
                epochs=set_epochs,
                regression=True,
                # classification=True,
                model_path=model_paths[dim],
                save_model_path=model_paths[dim],
                # current_dim=dim,
                current_dim=dim,
                test_path=te_path,
                test_label_path=telabels_path,
                nullset_path=te_path,
                # nullset_labels=null_arr[:, :, jj],
                # learning_rate=learn_rate,
                # dense_out=1,
                # hidden=30,
                # dense_in=10,
                # drop_per=.1,
                # normalize=False
            )
            resp = AFF.predict()

            try:
                respred[:, dim] = resp[:, 0]
            except:
                respred = np.empty((resp.shape[0], output_dimension))
                respred[:, dim] = resp[:, 0]
        print(f'0 dim prediction \n {respred[:, 0]}')
        dict4['prediction'] = respred
        graph_dict[param_pred] = dict4

        with open(
                r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_30dims\results_3dims.pkl',
                'wb') as f11:
            pickle.dump(graph_dict, f11)




        # # training and saving
        # AFF.mod_labels() # only for regression
        # history = AFF.train()
        # print(
        #     'hello world'
        # )
        # AFF.save_model()
        #
        # param = 'ff_100ep_train_ricoword_BOWavg_class'
        # AFF.training_results(param,
        #                     graph_dict_path= r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Atom-FFNN\graphing_data.pkl',
        #                     save_file=r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Atom-FFNN\graphing_data.pkl')
        #
        #
        #
        # # testing and saving probabilistic
        #
        # param = 'ff_100ep_nullset_ricoword_BOWavg_class'
        # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Atom-FFNN\graphing_data.pkl', 'rb') as f1:
        #     graph_dict = pickle.load(f1)
        # AFF.test_classification_results(graph_dict, param)
        #
        #
        # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Atom-FFNN\graphing_data.pkl', 'wb') as f2:
        #     pickle.dump(graph_dict, f2)
        #
        #
        # # testing and saving regression
        #
        #
        # param = 'ff_150ep_train_ricoword_BOWsum_proj'
        # AFF.training_results(param,
        #                     graph_dict_path= r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Metastimuli-Learn\Atom-FFNN\regress_graph_data.pkl',
        #                     save_file=r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Metastimuli-Learn\Atom-FFNN\regress_graph_data.pkl')
        #
        # param = 'ff_150ep_test_ricoword_BOWsum_proj'
        # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Metastimuli-Learn\Atom-FFNN\regress_graph_data.pkl', 'rb') as f3:
        #     regress_graph_dict = pickle.load(f3)
        # #
        # regress_graph_dict = AFF.test_regression_results(regress_graph_dict, param)
        # #
        # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Metastimuli-Learn\Atom-FFNN\regress_graph_data.pkl', 'wb') as f4:
        #     pickle.dump(regress_graph_dict, f4)


        # use case








