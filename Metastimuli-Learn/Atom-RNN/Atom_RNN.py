import os
import pickle
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import plot_model
from matplotlib import pyplot as plt

from kerastuner.tuners import RandomSearch
from kerastuner.tuners import BayesianOptimization
from kerastuner.tuners import Hyperband

class Atom_RNN():

    def __init__(self, data_path='', train_label_path='', test_path='',  test_label_path='',
                 nullset_path = '', nullset_labels_path='',
                 model_path = '', save_model_path='', trbatch_size=10, tebatch_size=1, epochs=100, curdim=0, input_size=10,
                  drop_per=0.1, units=100, output_size=1, steps=5, learn_rate=0.005, seed = 24,
                 regression=False, classification=False, optimize=False):
        try:
            with open(data_path, 'rb') as data_file:
                self.data = pickle.load(data_file)
        except ValueError:
            print('Data path does not exist')

        if isinstance(self.data, list):
            self.data = self.data_to_numpy(self.data, self.input_size)


        self.trbatch_size = trbatch_size
        self.tebatch_size = tebatch_size
        self.epochs = epochs
        self.drop_per = drop_per
        self.learn_rate = learn_rate
        self.steps = steps
        # self.input_shape = input_shape
        self.units = units
        self.output_size = output_size
        self.input_size = input_size
        self.curdim = curdim
        self.seed = seed
        if test_path != '':

            with open(test_path, 'rb') as test_file:
                self.test_data = pickle.load(test_file)
            if isinstance(self.test_data, list):
                self.test_data = self.data_to_numpy(self.test_data, self.input_size)


            with open(test_label_path, 'rb') as test_label_file:
                self.test_labels = pickle.load(test_label_file)

            self.test_labels = np.asarray(self.test_labels)
            if isinstance(self.test_labels, list):
                self.test_labels = self.data_to_numpy(self.test_labels, self.output_size)

            self.test_labels = self.splice_labels(self.test_labels, self.curdim)


        if train_label_path != '':
            with open(train_label_path, 'rb') as train_label_file:
                self.train_labels = pickle.load(train_label_file)
            self.train_labels = np.asarray(self.train_labels)
            if isinstance(self.train_labels, list):
                self.train_labels = self.data_to_numpy(self.train_labels, self.output_size)
            self.train_labels = self.splice_labels(self.train_labels, self.curdim)


        if nullset_path != '':
            # nullset is either train set, test set, or validation set
            with open(nullset_path, 'rb') as null_file:
                self.nullset = pickle.load(null_file)
            if not isinstance(self.nullset, np.ndarray):
                self.nullset = self.data_to_numpy(self.nullset, self.input_size)

            # nullset is the randomized labels of the previous dataset

            with open(nullset_labels_path, 'rb') as null_labels_file:
                self.nullset_labels = pickle.load(null_labels_file)
            if not isinstance(self.nullset_labels, np.ndarray):
                self.nullset_labels = self.data_to_numpy(self.nullset_labels, self.output_size)


        # self.data = self.data.T
        self.data = self.stepify(self.data, boollabels=False)
        self.train_labels = self.stepify(self.train_labels, boollabels=True)
        try:
            self.test_data = self.stepify(self.test_data, boollabels=False)
        except:
            self.test_data = None


        try:
            self.test_labels = self.stepify(self.test_labels, boollabels=True)
        except:
            self.test_labels = None


        self.history = dict()

        if not optimize:
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

    def __converttomatrix(self, data, labels=False):
        X, Y = [], []
        if labels:
            for ii in range(data.shape[0] - self.steps):
                d = ii + self.steps
                Y.append(data[d])
            return np.array(Y)
        else:
            for ii in range(data.shape[0] - self.steps):
                d = ii + self.steps
                X.append(data[ii:d, :])
            return np.array(X)



    def splice_labels(self, labels, dim):
        try:
            labs = labels[:, dim]
        except:
            labs = None
        return labs

    def stepify(self, arr, boollabels=False):
        # step_arr = np.empty((arr.shape[1]))
        if boollabels:
            for ii in range(self.steps):
                newrow = arr[-1]
                # newrow = np.reshape(newrow, (1, arr.shape[1]))
                arr = np.append(arr, newrow)

            # arr = (4__, 30, self.step)
            new_arr = self.__converttomatrix(arr, labels=boollabels)
            # new_arr = np.reshape(new_arr, (new_arr.shape[0], 1))
        else:
            for ii in range(self.steps):
                newrow = arr[-1, :]
                newrow = np.reshape(newrow, (1, arr.shape[1]))
                arr = np.append(arr, newrow, axis=0)

            # arr = (4__, 30, self.step)
            new_arr = self.__converttomatrix(arr, labels=boollabels)
            new_arr = np.reshape(new_arr, (new_arr.shape[0], new_arr.shape[2], new_arr.shape[1]))
        return new_arr





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


    def build_regression_model(self):
        winit = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=self.seed)
        binit = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=self.seed)
        model = keras.Sequential([
            # layers.Dense(30, input_shape=(30,self.steps)),
            layers.SimpleRNN(
                units=240,
                input_shape=(30, self.steps),
                # batch_size=self.batch_size,
                kernel_initializer=winit,
                bias_initializer=binit,
                activation='tanh'
                # dropout=self.drop_per,
                # input_shape=self.input_shape
            ),
            layers.Dense(111, activation='tanh'),
            layers.Dense(self.output_size,
                         activation='linear')
        ])
        sgd = keras.optimizers.SGD(learning_rate=self.learn_rate)
        model.compile(

            optimizer=sgd,
            loss=tf.keras.losses.mean_squared_error,
            metrics=['mean_squared_error']

        )
        return model

    def build_classification_model(self):
        winit = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=self.seed)
        binit = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=self.seed)
        model = keras.Sequential([
            layers.SimpleRNN(
                units=self.units,
                # dropout=self.drop_per,
                # input_shape=self.input_shape,
            kernel_initializer = winit, bias_initializer = binit
            ),
            layers.Dense(self.output_size,
                         activation='softmax', kernel_initializer=winit, bias_initializer=binit)
        ])

        model.compile(

            optimizer='adam',
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=['accuracy']

        )
        return model
    def train(self):

        self.history = self.model.fit(
            self.data, self.train_labels, epochs=self.epochs
            # batch_size=self.trbatch_size
            # , shuffle=False
            # shuffles batches
            , verbose=1  # supresses progress bar
            )
        # except ValueError:
        #     self.data = self.data.T
        #     self.history = self.model.fit(
        #         self.data, self.train_labels, epochs=self.epochs, batch_size=self.batch_size, shuffle=False
        #         # shuffles batches
        #         , verbose=1  # supresses progress bar
        #     )
        return self.history


    def test(self):

        result = self.model.evaluate(self.test_data, self.test_labels
                                     # , batch_size=self.tebatch_size
                                     , verbose=1
                                     )

        # except:
        #     self.test_data = self.test_data.T # transpose data
        #     result = self.model.evaluate(self.test_data, self.test_labels, batch_size=self.batch_size)

        if len(result) > 2:
            print(f'Test results: Loss= {result[0]:.3f}, Accuracy = {100*result[1]:.3f}%')
            return result[0], result[1]
        else:
            print(f'Test results: Loss= {result[0]:.3f}')
            return result[0]


    def predict(self):
        # self.data = self.data.T

        result = self.model.predict(self.data)
        return result


    def save_model(self):

        keras.models.save_model(
            self.model, self.save_model_path
        )



    def build_model(self, hp):
        model = keras.Sequential()

        model.add(layers.SimpleRNN(
            hp.Int('rnn_units', min_value=30, max_value=900, step=15),
            input_shape=(30, self.steps),
            activation=hp.Choice('rnn_activation', values=['tanh', 'sigmoid', 'softmax'])
        ))

        for ii in range(hp.Int('num_h-layers', 1, 10)):
            model.add(
                layers.Dense(
                    hp.Int(f'Dense_{ii}_units', min_value=16, max_value=1600, step=16),
                    activation=hp.Choice(f'Dense_{ii}_activation', values=['tanh', 'sigmoid', 'softmax'])
                )
            )

        model.add(layers.Dense(1, activation='linear'))

        # keras.optimizers.SGD(learning_rate=1e-2, momentum=0, nesterov=False)
        # keras.optimizers.Adagrad(leaning_rate=___, initial_accumulator_value=1e-1, epsilon=1e-7)
        # keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False)
        # keras.optimizers.RMSprop(learning_rate=1e-3, rho=0.9, momentum=0.0, epsilon=1e-7, centered=False)
        # keras.optimizers.Adadelta(learning_rate=1e-3, rho=0.95, epsilon=1e-7)
        # keras.optimizers.Adamax(learning_rate=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-7) may be superior with embeddings



        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp.Float('learning_rate',
                                                                   min_value=1e-5,
                                                                   max_value=1e-2,
                                                                   sampling='LOG',
                                                                   ),
                                            beta_1=hp.Float('beta_1', min_value=0.7, max_value=0.95, step=5e-2),
                                            beta_2=hp.Float('beta_2', min_value=0.99, max_value=0.9999, step=9e-4),

                                            ),

            loss=tf.keras.losses.mean_squared_error,
            metrics=['mean_squared_error']
        )
        return model


    def random_search(self):
        tuner = RandomSearch(
            self.build_model,
            objective='val_mean_squared_error',
            max_trials=1,
            executions_per_trial=3,
            # directory=dir,
            )

        tuner.search(
            x = self.data,
            y = self.train_labels,
            epochs = self.epochs,
            # batch_size = self.batch_size,
            validation_data=(self.test_data, self.test_labels)

        )
        return tuner

    def bayesian(self):
        tuner = BayesianOptimization(
            self.build_model,
            objective='val_mean_squared_error',
            max_trials=3,
            num_initial_points=3,
            seed=self.seed,
            # directory=dir,
            )

        tuner.search(
            x = self.data,
            y = self.train_labels,
            epochs = self.epochs,
            # batch_size = self.batch_size,
            validation_data=(self.test_data, self.test_labels)
        )
        return tuner

    def hyperband(self):
        tuner = Hyperband(
            self.build_model,
            objective='val_mean_squared_error',
            max_epochs=self.epochs+10,
            factor=3,
            hyperband_iterations= 5, # The number of times to iterate over the full Hyperband algorithm. It is recommended to set this to as high a value as is within your resource budget.
            # directory=dir,
            seed=self.seed
            )

        tuner.search(
            x = self.data,
            y = self.train_labels,
            epochs = self.epochs,
            # batch_size = self.batch_size,
            validation_data=(self.test_data, self.test_labels)
        )

        return tuner



if __name__ == '__main__':
    with tf.device('/cpu:0'):
        trbatch_size = 5
        tebatch_size = 1
        inner_epochs = 1
        outer_epochs = 500
        numsteps = 5
        epochs = outer_epochs

        learn_rate = 0.005
        # curdim = 0
        model_paths = [
            r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\BOWsum\w100\rnn_model500_3dims_rnntanh240-tanh111_00',
            r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\BOWsum\w100\rnn_model500_3dims_rnntanh240-tanh111_01',
            r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\BOWsum\w100\rnn_model500_3dims_rnntanh240-tanh111_02'
        ]

        with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\results_3dims.pkl', 'rb') as f10:
            graph_dict = pickle.load(f10)
        # graph_dict = dict()

        output_dimension = len(model_paths) # total output dimension
        tr_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\BOWsum\w100\train.pkl'
        trlabels_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\BOWsum\w100\train_labels.pkl'

        te_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\BOWsum\w100\test.pkl'
        telabels_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\BOWsum\w100\test_labels.pkl'

        restr = np.empty((output_dimension, epochs))
        reste = np.empty((output_dimension, epochs))
        # null = np.empty((output_dimension, epochs, null_arr.shape[2]))
        respred = None
        null_holder = []
        dict1 = dict()
        dict2 = dict()
        dict3 = dict()
        dict4 = dict()
        for dim in range(output_dimension):
            RNN = Atom_RNN(
                data_path=tr_path
                , train_label_path=trlabels_path
                , test_path=te_path
                , test_label_path=telabels_path
                # , model_path=r''
                , save_model_path=model_paths[dim]
                , trbatch_size=trbatch_size
                , epochs=inner_epochs
                # , drop_per=0.1
                , steps = numsteps
                , regression=True
                # , input_shape=(454,30)
                , output_size=1
                # , units=454

            )
            RNN.save_model()
            for ep in range(epochs):
                RNN = Atom_RNN(
                    data_path=tr_path
                    , train_label_path=trlabels_path
                    , test_path=te_path
                    , test_label_path=telabels_path
                    , model_path=model_paths[dim]
                    , save_model_path=model_paths[dim]
                    , trbatch_size=trbatch_size
                    , epochs=inner_epochs
                    # , drop_per=0.1
                    , regression=True
                    , steps=numsteps
                    # , input_shape=(454,30)
                    , output_size=1
                    # , units=454

                               )

                print(f'Epoch: {ep}')
                history = RNN.train()
                trloss = history.history['loss']
                restr[dim, ep] = trloss[0]
                RNN.save_model()
                reste[dim, ep] = RNN.test()
                keras.backend.clear_session()
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
                #                 , steps = numsteps
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
            # null_avg = np.average(null, axis=2)

            param_tr = f'rnn_500ep_train_rico_BOWsum_w_100_3dims_rnntanh240-tanh111_5st'
            param_te = f'rnn_500ep_test_rico_BOWsum_w_100_3dims_rnntanh240-tanh111_5st'
            # param_null = 'ff_50ep_nullset_rico10dims_ndelta_w_500_3dim_tanh_02'


            dict1['loss'] = restr
            dict1['epochs'] = epochs
            graph_dict[param_tr] = dict1

            dict2['loss'] = reste
            dict2['epochs'] = epochs
            graph_dict[param_te] = dict2

            # dict3['loss'] = null_avg
            # dict3['epochs'] = epochs
            # graph_dict[param_null] = dict3

        for dim in range(output_dimension):
            param_pred = f'rnn_500ep_pred_rico_BOWsum_w_100_3dims_rnntanh240-tanh111_5st'
            AFF = Atom_RNN(
                data_path=tr_path,
                # local atom vector path
                train_label_path=trlabels_path,
                trbatch_size=trbatch_size,
                epochs=inner_epochs,
                regression=True,
                # classification=True,
                model_path=model_paths[dim],
                save_model_path=model_paths[dim],
                steps=numsteps,
                # current_dim=dim,
                # current_dim=dim,
                # test_path=te_path,
                # test_label_path=telabels_path,
                # nullset_path=te_path,
                # nullset_labels=null_arr[:, :, jj],
                # learning_rate=learn_rate,
                # dense_out=1,
                # hidden=30,
                # dense_in=10,
                # drop_per=.1,
                # normalize=False
            )
            resp = AFF.predict()
            resp = np.reshape(resp, (len(resp), 1))
            try:
                if respred == None:
                    respred = np.empty((len(resp), output_dimension))
            except:
                pass
            respred[:, dim] = resp[:, 0]

            print(resp)

        dict4['prediction'] = respred
        graph_dict[param_pred] = dict4

        with open(
                r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\results_3dims.pkl',
                'wb') as f11:
            pickle.dump(graph_dict, f11)

    #     self, data_path = '', train_label_path = '', test_path = '', test_label_path = '',
    #     model_path = '', save_model_path = '', batch_size = 10, epochs = 100,
    #     drop_per = 0.1, dense_out = 2, dense_in = 2, hidden = 50,
    #     regression = False, classification = False):