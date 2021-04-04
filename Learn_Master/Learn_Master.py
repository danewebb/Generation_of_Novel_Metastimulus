import sys
sys.path.append("..") # Adds higher directory to python modules path.
import os
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
up_dir = os.path.join(script_dir, '..')
import time
# from atom_tag_pairing import Atom_Tag_Pairing


from Lib.Atom_Embeddings.atom_embedding import Atom_Embedder
from Lib.Metastimuli_Learn.Atom_FFNN.Atom_FFNN import Atom_FFNN
from Lib.Metastimuli_Learn.Atom_RNN.Atom_RNN import Atom_RNN
from Lib.Word_Embeddings.Weighting_Keywords import Weighting_Keyword
# fromword_embedding_ricocorpus import word_embedding_ricocorpus
# from word_embedding import Sciart_Word_Embedding
from Lib.Shuffled_Data_1.Randomizer import Shuffler


# from Atom_Embeddings.atom_embedding import Atom_Embedder
# from Metastimuli_Learn.Atom_FFNN.Atom_FFNN import Atom_FFNN
# from Metastimuli_Learn.Atom_RNN.Atom_RNN import Atom_RNN
# from Word_Embeddings.Weighting_Keywords import Weighting_Keyword
# from Shuffled_Data_1.Randomizer import Shuffler



# from pathlib import Path
# from Process_sciart import Process_Sciart
# from Label_Text_Builder import Label_Text_Builder
# from Processing import Tex_Processing

# PIMS filter import
import numpy as np
import pickle
# import tensorflow as tf
# from tensorflow import keras
import keras
# import keras.layers as layers

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Learn_Master():
    """

    """
    def __init__(self, dataset_dict, vocab, ordered_encdata, ordered_labels, input_dims, we_models, output_dims, keyword_weight_factor,
                 optimizers, atom_methods, nn_architectures, hyperoptimizers, ints, projections=None,
                 savepath_model = None, fitness_savepath='fitness_savepath',
                 NN_prebuilt_model=None, NNclasse=None, kt_master_dir='untitled',
                 WE_classes=None, vocab_save_path='', data_save_path='', checkpoint_filepath='untitled_checkpoints',
                 projection=None, adjacency=None, curout_dim=0, raw_paras=None,
                 epochs=1, fitness_epochs=10, rnn_steps=3,

                 maxiter=100, mindiff=1e-6, mu0=4, alpha=1, delta=4,
                 ):

        """

        :param dataset_dict: dictionary with dataset information. See Label_Text_Builder.py
        :param vocab: list of word embedding vocab
        :param ordered_encdata: list of encoded datasets used in the word embeddings
        :param ordered_labels: list of labels aligned with the ordered_encdata
        :param input_dims: list of input dimesnions
        :param we_models: list of trained word embedding models (keras)
        :param output_dims: list of projection output dimensions
        :param keyword_weight_factor: list of values to weight keywords
        :param optimizers: list of gradient optimizers ['sgd', 'adamax', 'adagrad', 'adam', 'adadelta', 'rmsprop']
        :param atom_methods: list of atom methods ['bowsum', 'bowavg', 'ndelta']
        :param nn_architectures: list of NN architectures ['ff', 'rnn']
        :param hyperoptimizers: list of hyperparameter optimizer methods ['random', 'bayes', 'hyper']
        :param ints: list of the indices that are integers in the objective variables
        :param savepath_model:
        :param NN_prebuilt_model:
        :param NNclasse:
        :param WE_classes:
        :param vocab_save_path:
        :param data_save_path:
        :param projection:
        :param adjacency:
        :param curout_dim:
        :param epochs:
        :param rnn_steps:
        :param nvar:
        :param maxiter:
        :param mindiff:
        :param mu0:
        :param alpha:
        :param delta:
        """


        """
        let us assume dataset is already created and encoded.

        objectives = [_, input dims / model, output dims / labels,
              key weight, atom method, optimizer, NN arch, hyperoptimizers]
        """

        # [0, 4, 2, 0, 2, 1, 0, 2]
        # [~, 20, 4, 1, ndelta, adam, ff, hyperband]


        self.lb = [0, 0, 0, 0, 0, 0, 0, 0]
        self.ub = [2, len(input_dims), len(output_dims), len(keyword_weight_factor),
                   len(atom_methods), len(optimizers), len(nn_architectures), len(hyperoptimizers)]
        if len(self.lb) != len(self.ub):
            raise ValueError('lb and ub must be the same lengths')

        self.nvar = len(self.lb)

        self.checkpoint_filepath = checkpoint_filepath
        self.dataset_dict = dataset_dict
        self.vocab = vocab
        self.ordered_encdata = ordered_encdata
        self.ordered_labels = ordered_labels
        self.ints = ints
        self.keyword_weigth_factor = keyword_weight_factor

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.curout_dim = curout_dim
        self.optimizers = optimizers
        self.nn_architectures = nn_architectures
        self.atom_methods = atom_methods
        self.hyperoptimizers = hyperoptimizers
        self.alpha = alpha
        self.delta = delta
        # if adjacency is not None:
        #     self.adj = adjacency
        # if projection is not None:
        #     self.proj = projection
        # if projection is not None and adjacency is not None:
        #     self.label_pairing()

        self.epochs = epochs
        self.we_models = we_models
        self.rnn_steps = rnn_steps
        self.fitness_epochs = fitness_epochs
        self.mindiff = mindiff
        self.maxiter = maxiter
        self.mu0= mu0
        self.mu = mu0
        self.train_for = 0
        self.n_nulls = 0
        self.npop = 0


        self.saver = dict() # Saves status and variables for pattern search
        self.fitness_savepath = fitness_savepath

        if raw_paras is not None:
            self.raw_paras = raw_paras

        if savepath_model is not None:
            self.savepath_model = savepath_model
        else:
            self.savepath_model = "optimized_models/untitled_model_dim{dim}".format(dim = self.curout_dim)
            # self.savepath_model = "optimized_models/untitled_model_dim"

        if projections is not None:
            self.projections = projections

        # need to define later
        self.list_train_paths = None
        self.list_train_label_paths = None
        self.list_test_paths = None
        self.list_test_label_paths = None

        self.ordtrain = None; self.ordtrain_labels = None; self.ordtest = None; self.ordtest_labels = None
        self.train = None; self.train_labels = None; self.test = None; self.test_labels = None

        # self.kt_dir = dict()
        self.kt_master_dir = kt_master_dir


        if os.path.exists(kt_master_dir + '/fits.pkl'):
            with open(kt_master_dir + '/fits.pkl', 'rb') as f:
                self.kt_fits = pickle.load(f)
            with open(kt_master_dir + '/tuners.pkl', 'rb') as f:
                self.trained_tuners = pickle.load(f)
        else:
            self.kt_fits = dict()
            self.trained_tuners = dict()
    # def label_pairing(self):
    #     from atom_tag_pairing import Atom_Tag_Pairing
    #
    #     if self.adj is not None:
    #         ATP = Atom_Tag_Pairing(self.dataset, adjacency=self.adj, projection=self.proj)
    #     else:
    #         ATP = Atom_Tag_Pairing(self.dataset, projection=self.proj)
    #     ATP.tag_pairing()
    #     self.labels = ATP.projection_vectors()

    def build_objdataset(self, objectives):
        """

        :param objectives:
        :return:
        """


        if objectives.ndim == 1:
            obj = objectives.astype(int)
            obj = np.reshape(obj, (1, len(obj)))
        else:
            obj = objectives

        if obj[0, 1] == 8:
            # sciart
            voc = self.vocab[1]
            enc = self.ordered_encdata[1]
        else:
            # rico WE
            voc = self.vocab[0]
            enc = self.ordered_encdata[0]


        input_dimension = self.input_dims[obj[0, 1]]
        we_model = keras.models.load_model(self.we_models[obj[0, 1]])

        output_dimension = self.output_dims[obj[0, 2]]
        labels = self.ordered_labels[obj[0, 2]]

        # encoded, nlabels = self.eliminate_empties(enc, labels)

        key_weight = self.keyword_weigth_factor[obj[0, 3]]

        atom_method = self.atom_methods[obj[0, 4]]
        # atom_method = self.atom_methods[3]

        atom_vecs = self.atom_embedding(atom_method, we_model, voc, key_weight, enc)
        self.split_shuff(atom_vecs, labels, input_dimension)

        # self.kt_dir[str(obj)] = self.kt_master_dir + '/' + str(obj)



        return obj


    def eliminate_empties(self, enc, labs):
        # just eliminates empty paragraphs and their labels
        atoms = []
        labels = []
        for para, lab in zip(enc, labs):
            if para == [0] or para == [] or para == None:
                pass

            else:
                atoms.append(para)
                labels.append(lab)

        return atoms, labels


    def atom_embedding(self, atom_embed_method, we_model, vocab, key_weight, ordered_encdata, ndel=8):
        atom_vecs = []

        AE = Atom_Embedder(we_model.layers[0].get_weights()[0], vocab)
        WK = Weighting_Keyword(vocab, key_weight)
        WK.keywords_in_vocab()


        if atom_embed_method == 'sum_atoms':
            for ii, para in enumerate(ordered_encdata):
                if para:
                    weights = WK.keyword_search(para)
                    atom_vecs.append(AE.sum_atoms(para, weights=weights))
                else:

                    p = [np.random.randint(0, len(vocab)), np.random.randint(0, len(vocab)), np.random.randint(0, len(vocab))]
                    atom_vecs.append(AE.sum_atoms(p))
        elif atom_embed_method == 'avg_atoms':
            for ii, para in enumerate(ordered_encdata):
                if para:
                    weights = WK.keyword_search(para)
                    atom_vecs.append(AE.avg_atoms(para, weights=weights))
                else:
                    p = [np.random.randint(0, len(vocab)), np.random.randint(0, len(vocab)), np.random.randint(0, len(vocab))]
                    atom_vecs.append(AE.avg_atoms(p))
        elif atom_embed_method == 'ndelta':
            for ii, para in enumerate(ordered_encdata):
                if para:
                    weights = WK.keyword_search(para)
                    atom_vecs.append(AE.sum_of_ndelta(para, ndel, weights=weights))
                else:
                    p = [np.random.randint(0, len(vocab)), np.random.randint(0, len(vocab)), np.random.randint(0, len(vocab))]
                    atom_vecs.append(AE.sum_of_ndelta(p, ndel))

        elif atom_embed_method == 'pvdm':
            AE.pvdm_train(self.raw_paras)
            for para, rpara in zip(ordered_encdata, self.raw_paras):
                if para:
                    weights = WK.keyword_search(para)
                    sweight = sum(weights)
                    atom_vecs.append(AE.pvdm(rpara, sweight))
                else:
                    # a= np.zeros((atom_vecs[-1].shape))
                    atom_vecs.append(np.zeros(atom_vecs[-1].shape))


        # with open(Path(f'C://Users//liqui//PycharmProjects//Generation_of_Novel_Metastimulus///Atom-Embeddings//'
        #                f'embeddings//{we_embed}_w{self.keyword_weigth_factor}_in{atom_vecs[0].shape[0]}_{atom_embed_method}.pkl'),
        #           'wb') as f:
        #     pickle.dump(atom_vecs, f)

        return atom_vecs


    def split_shuff(self, atom_vecs, labels, input_dims):

        SH = Shuffler(atom_vecs, labels, input_dims)

        self.ordtrain, self.ordtrain_labels, self.ordtest, self.ordtest_labels = SH.train_test_split()
        self.train, self.train_labels, self.test, self.test_labels = SH.shuff_train_test_split()


    def penalties(self, loss, wkey, out):
        if wkey > 1:
            kpen = 0.0001 * wkey
            kpen = kpen + 1
        else:
            kpen = 1

        if out > 2:
            open = 0.001*out
            open = open + 1
        else:
            open = 1

        penloss = loss*kpen*open
        return penloss


    def ff_fitness(self, input_dim, out_dims, kweight, obj, optimizer='sgd', nn_architecture='ff', hyperoptimizer='random'):
        total_loss = []
        all_models = []
        all_tuners = []

        # kt_dir = self.kt_dir[str(obj)]


        if str(obj) in self.kt_fits.keys():
            return self.kt_fits[str(obj)], self.trained_tuners[str(obj)]
        # with tf.device('/cpu:0'):
        for dim in range(out_dims):
            if nn_architecture == 'ff':
                AFF = Atom_FFNN(
                    data=self.train,
                    train_labels=self.train_labels,
                    test=self.test,
                    test_labels=self.test_labels,
                    nullset=self.test,
                    # batch_size=set_batch_size,
                    # epochs=set_epochs,
                    # learning_rate=learn_rate,
                    dense_out=1,
                    dense_in=input_dim,
                    current_dim=dim,
                    optimize=True,
                    seed=24,
                    optimizer=optimizer,
                    # kt_directory=kt_dir,
                    epochs=self.fitness_epochs,
                    initial_points=3,
                    hyper_maxepochs=3,
                    hyper_iters=3
                )
                xtest = AFF.test_data
                ytest = AFF.test_labels

                if hyperoptimizer == 'random':
                    tuner = AFF.random_search()
                    best_model = tuner.get_best_models(num_models=1)[0]
                    loss = best_model.evaluate(xtest, ytest)  # list[mse loss, mse]
                    print('Completed random search/n')
                elif hyperoptimizer == 'bayes':
                    tuner = AFF.bayesian()
                    best_model = tuner.get_best_models(num_models=1)[0]
                    loss = best_model.evaluate(xtest, ytest)  # list[mse loss, mse]
                    print('Completed bayesian/n')
                elif hyperoptimizer == 'hyper':
                    tuner = AFF.hyperband()
                    best_model = tuner.get_best_models(num_models=1)[0]
                    loss = best_model.evaluate(xtest, ytest)  # list[mse loss, mse]
                    print('Completed hyperband/n')



            else:
                RNN = Atom_RNN(
                    data=self.ordtrain,
                    train_labels=self.ordtrain_labels,
                    test=self.ordtest,
                    test_labels=self.ordtest_labels,
                    nullset=self.ordtest,
                    # batch_size=set_batch_size,
                    # epochs=set_epochs,
                    # learning_rate=learn_rate,
                    output_size=1,
                    input_size=input_dim,
                    curdim=dim,
                    optimize=True,
                    seed=24,
                    optimizer=optimizer,
                    # kt_directory=kt_dir,
                    steps=self.rnn_steps,
                    epochs=self.fitness_epochs,
                    initial_points=3,
                    hyper_maxepochs=3,
                    hyper_iters=3
                )
                xtest = RNN.test_data
                ytest = RNN.test_labels

                if hyperoptimizer == 'random':
                    tuner = RNN.random_search()
                    best_model = tuner.get_best_models(num_models=1)[0]
                    loss = best_model.evaluate(xtest, ytest)  # list[mse loss, mse]
                    print('Completed random search/n')
                elif hyperoptimizer == 'bayes':
                    tuner = RNN.bayesian()
                    best_model = tuner.get_best_models(num_models=1)[0]
                    loss = best_model.evaluate(xtest, ytest)  # list[mse loss, mse]
                    print('Completed bayesian/n')
                elif hyperoptimizer == 'hyper':
                    tuner = RNN.hyperband()
                    best_model = tuner.get_best_models(num_models=1)[0]
                    loss = best_model.evaluate(xtest, ytest)  # list[mse loss, mse]
                    print('Completed hyperband/n')

            total_loss.append(loss[0])
            all_models.append(best_model)
            all_tuners.append(tuner)



        avgloss = sum(total_loss)/len(total_loss)
        # penloss = self.penalties(avgloss, kweight, out_dims)
        penloss = avgloss


        self.kt_fits[str(obj)] = penloss
        self.trained_tuners[str(obj)] = all_tuners
        # return penloss, all_models, all_tuners
        return penloss, all_tuners



    def make_mesh(self, x):
        jj = 0
        mesh = np.zeros((2*self.nvar, self.nvar))
        gps = np.zeros((2 * self.nvar, self.nvar))

        for ii in range(mesh.shape[0]):
            if ii == self.nvar:
                jj = 0
            if ii < self.nvar:
                mesh[ii, jj] = self.mu
                jj += 1

            else:
                mesh[ii, jj] = -self.mu
                jj += 1

        for ii in range(mesh.shape[0]):
            gps[ii, :] = mesh[ii, :] + x

        return gps


    def intcons(self, x):
        xnew = x
        for ii in range(x.shape[0]):
            for int_idx in self.ints:
                xnew[ii, int_idx] = round(x[ii, int_idx])
                if xnew[ii, int_idx] < self.lb[int_idx]:
                    xnew[ii, int_idx] = self.lb[int_idx]
                elif xnew[ii, int_idx] > self.ub[int_idx]:
                    xnew[ii, int_idx] = self.ub[int_idx]

        return xnew.astype(int)


    def enforce_bounds(self, x):
        if x.ndim == 1:
            x = np.reshape(x, (1, len(x)))
        xnew = np.zeros(x.shape)
        for jj in range(x.shape[1]):
            for ii in range(x.shape[0]):
                if x[ii, jj] < self.lb[jj]:
                    xnew[ii, jj] = self.lb[jj]
                elif x[ii, jj] >= self.ub[jj]:
                    xnew[ii, jj] = self.ub[jj]-1
                else:
                    xnew[ii, jj] = x[ii, jj]

        return xnew

    def save_fitness(self, fitness):
        try:
            with open(self.fitness_savepath, 'rb') as f:
                fit_res = pickle.load(f)

        except:
            fit_res = []


        fit_res.append(fitness)
        with open(self.fitness_savepath, 'wb') as f:
            pickle.dump(fit_res, f)






    def phase0(self):
        master_dict = self.saver['master_dict']
        dict_template = self.saver['dict_template']

        count = 0
        obj = np.zeros((1, self.nvar), dtype='int32')
        for ii in range(self.nvar):
            obj[0, ii] = np.random.randint(self.lb[ii], self.ub[ii])

        objold = np.zeros((1, self.nvar), dtype='int32')
        objnew = np.zeros((1, self.nvar), dtype='int32')
        mu = self.mu0

        oldfit = np.inf
        newfit = 0
        obj = self.enforce_bounds(obj)
        obj = self.intcons(obj)
        obj = self.build_objdataset(obj)

        curfit, best_models, best_tuners = self.ff_fitness(
            self.input_dims[obj[0, 1]],
            self.output_dims[obj[0, 2]],
            self.keyword_weigth_factor[obj[0, 3]],
            optimizer=self.optimizers[obj[0, 5]],
            nn_architecture=self.nn_architectures[obj[0, 6]],
            hyperoptimizer=self.hyperoptimizers[obj[0, 7]]
        )

        if self.train_for > 0:
            count += 1
            self.save_fitness(curfit)
            results = self.train_best_hps(best_tuners, obj, self.train_for, num_nulls=self.n_nulls)
            self.save_checkpoint_results(dict_template, count, results, obj, best_tuners, best=True)  # save rando gen
        iter = 0

        self.saver['master_dict'] = master_dict
        self.saver['obj'] = obj
        self.saver['dict_template'] = dict_template
        self.saver['count'] = count
        self.saver['curfit'] = curfit
        self.saver['tuners'] = best_tuners
        self.saver['mu'] = mu
        self.saver['iter'] = iter
        self.saver['newfit'] = newfit
        with open(self.saver['savepath'], 'wb') as f:
            pickle.dump(self.saver, f)

    # def phase1(self, iter, obj, count, curfit, master_dict, mu, newfit, dict_template):
    def phase1(self):
        iter = self.saver['iter']
        obj = self.saver['obj']
        count = self.saver['count']
        curfit = self.saver['curfit']
        master_dict = self.saver['master_dict']
        dict_template = self.saver['dict_template']
        mu = self.saver['mu']
        newfit = self.saver['newfit']
        dict_template = self.saver['dict_template']


        changeflag = 0
        while iter < self.maxiter:
            if iter % round(self.maxiter/2):
                t = time.localtime()
                now = time.strftime("%H:%M:%S", t)
                print('-----------------------------------------------------------------------------------------------')
                print('-----------------------------------------------------------------------------------------------')
                print('-----------------------------------------------------------------------------------------------')
                print('Begin iteration {i} at {n}'.format(i=iter, n=now))
                print('-----------------------------------------------------------------------------------------------')
                print('-----------------------------------------------------------------------------------------------')
                print('-----------------------------------------------------------------------------------------------')

            mesh = self.make_mesh(obj)
            mesh = self.enforce_bounds(mesh)
            mesh = self.intcons(mesh)

            # check if any of the exploratory points have lower fitness
            for ii in range(0, 2 * self.nvar):
                mesh[ii, :] = self.build_objdataset(mesh[ii, :])
                print(mesh[ii, :])
                # meshfit = self.gen_random_loss()
                meshfit, meshmods, tuners = self.ff_fitness(
                    self.input_dims[mesh[ii, 1]],
                    self.output_dims[mesh[ii, 2]],
                    self.keyword_weigth_factor[mesh[ii, 3]],
                    optimizer=self.optimizers[mesh[ii, 5]],
                    nn_architecture=self.nn_architectures[mesh[ii, 6]],
                    hyperoptimizer=self.hyperoptimizers[mesh[ii, 7]]
                )

                if self.train_for > 0:
                    count += 1
                    self.save_fitness(meshfit)
                    results = self.train_best_hps(tuners, mesh[ii, :], self.train_for, num_nulls=self.n_nulls)
                    self.save_checkpoint_results(dict_template, count, results, mesh[ii, :], tuners, best=False)

                if meshfit < curfit:
                    oldfit = curfit
                    curfit = meshfit

                    objnew = mesh[ii, :]
                    objnew = np.reshape(objnew, (1, len(objnew)))

                    best_model = meshmods
                    best_tuner = tuners
                    changeflag = 1
                    start = time.time()

                    if self.train_for > 0:
                        count += 1
                        self.save_fitness(curfit)
                        chosen_results = self.train_best_hps(best_tuner, objnew, self.train_for, num_nulls=self.n_nulls)
                        master_dict[dict_template + '{count}'.format(count=count)] = chosen_results
                        self.save_checkpoint_results(dict_template, count, chosen_results, objnew, best_tuner,
                                                     best=False)
                        bestcount = count

            if changeflag == 0:
                objnew = obj
                mu = mu - self.delta
                iter += 1
                self.saver['iter'] = iter
                self.saver['count'] = count
                self.saver['mu'] = mu

                with open(self.saver['savepath'], 'wb') as f:
                    pickle.dump(self.saver, f)



            elif changeflag == 1:
                changeflag = 0
                self.saver['iter'] = 0

                self.saver['phase'] = 2
                self.saver['objnew'] = objnew
                self.saver['tuners'] = best_tuner
                self.saver['count'] = count
                self.saver['curfit'] = curfit
                self.saver['bestcount'] = bestcount
                self.saver['start'] = start
                self.save_checkpoint_results(dict_template, bestcount, chosen_results, objnew, best_tuner, best=True)

                with open(self.saver['savepath'], 'wb') as f:
                    pickle.dump(self.saver, f)

                self.phase2()

                with open(self.saver['savepath'], 'rb') as f:
                    self.saver = pickle.load(f)

            else:
                raise ValueError('Changeflag must be either 0 or 1')


        # if changeflag == 1:

        # return objnew, best_tuner, count, bestcount, chosen_results, newfit, curfit, obj, master_dict


    # def phase2(self, objnew, best_tuner, count, bestcount, newfit, chosen_results, curfit, obj, master_dict):
    def phase2(self):

        objnew = self.saver['objnew']
        tuners = self.saver['tuners']
        dict_template = self.saver['dict_template']
        count = self.saver['count']
        bestcount = self.saver['bestcount']
        newfit = self.saver['newfit']
        curfit = self.saver['curfit']
        # chosen_results = self.saver['chosen_results']
        obj = self.saver['obj']
        master_dict = self.saver['master_dict']
        start = self.saver['start']
        # #####
        # count += 1
        # self.save_checkpoint_results(dict_template, count, curfit, objnew, best=True)
        changeflag = 0
        mu = self.mu0
        self.saver['mu'] = mu
        if newfit >= curfit:
            newfit = 0

        while newfit < curfit:
            objold = obj
            oldfit = curfit
            obj = objnew
            objnew = objold + self.alpha * (obj - objold)
            objnew = self.enforce_bounds(objnew)
            objnew = self.intcons(objnew)
            objnew = self.build_objdataset(objnew)

            # newfit = self.gen_random_loss()
            newfit, newmodels, new_tuners = self.ff_fitness(
                self.input_dims[objnew[0, 1]],
                self.output_dims[objnew[0, 2]],
                self.keyword_weigth_factor[objnew[0, 3]],
                optimizer=self.optimizers[objnew[0, 5]],
                nn_architecture=self.nn_architectures[objnew[0, 6]],
                hyperoptimizer=self.hyperoptimizers[objnew[0, 7]]
            )

            if self.train_for > 0:
                count += 1
                self.save_fitness(newfit)
                results = self.train_best_hps(new_tuners, objnew, self.train_for, num_nulls=self.n_nulls)
                master_dict[dict_template + '{count}'.format(count=count)] = results

                self.save_checkpoint_results(dict_template, count, results, objnew, new_tuners, best=False)

            #####
            # don't save here
            # count += 1
            # self.save_checkpoint_results(dict_template, count, newfit, obj, best=False)

            mesh = self.make_mesh(objnew)
            mesh = self.enforce_bounds(mesh)
            mesh = self.intcons(mesh)

            for ii in range(0, 2 * self.nvar):
                self.build_objdataset(mesh[ii, :])

                meshfit, meshmod, tuner = self.ff_fitness(

                    self.input_dims[mesh[ii, 1]],
                    self.output_dims[mesh[ii, 2]],
                    self.keyword_weigth_factor[mesh[ii, 3]],
                    optimizer=self.optimizers[mesh[ii, 5]],
                    nn_architecture=self.nn_architectures[mesh[ii, 6]],
                    hyperoptimizer=self.hyperoptimizers[mesh[ii, 7]]
                )
                if self.train_for > 0:
                    count += 1
                    self.save_fitness(meshfit)
                    results = self.train_best_hps(tuner, mesh[ii, :], self.train_for, num_nulls=self.n_nulls)
                    master_dict[dict_template + '{count}'.format(count=count)] = results

                    self.save_checkpoint_results(dict_template, count, results, mesh[ii, :], tuner, best=False)

                if meshfit < newfit:
                    oldfit = newfit
                    newfit = meshfit
                    objnew = mesh[ii, :]
                    best_model = meshmod
                    best_tuner = tuner
                    changeflag = 1

                    if self.train_for > 0:
                        count += 1
                        self.save_fitness(newfit)
                        chosen_results = self.train_best_hps(best_tuner, objnew, self.train_for, num_nulls=self.n_nulls)
                        self.save_checkpoint_results(dict_template, count, chosen_results, objnew, best_tuner,
                                                     best=False)
                        bestcount = count



            if changeflag == 1:
                curfit = newfit
                obj = objnew
                self.save_checkpoint_results(dict_template, bestcount, chosen_results, objnew, best_tuner,
                                             best=True)
                changeflag = 0
                self.saver['tuners'] = best_tuner
                if self.train_for > 0:
                    end = time.time()

                    # results = self.train_best_hps(best_tuner, objnew, train_for, num_nulls=n_nulls)
                    # master_dict[dict_template + '{count}'.format(count=count)] = results
                    # count += 1
                    # print(count)
                    # self.save_checkpoint_results(dict_template, count, results, objnew, best_tuner, best=False)

                    print('New center took {time} hours'.format(time=(end - start) / 3600))



            self.saver['objnew'] = objnew

            self.saver['obj'] = obj
            self.saver['count'] = count
            self.saver['newfit'] = newfit
            self.saver['curfit'] = curfit
            self.saver['master_dict'] = master_dict

            self.saver['phase'] = 1

            with open(self.saver['savepath'], 'wb') as f:
                pickle.dump(self.saver, f)




    def PS_integer(self, train_for=0, dict_template='untitled_dict', master_dict=None, n_nulls=50, saver_path='ps_saver_path'):
        self.train_for = train_for
        self.n_nulls = n_nulls


        if os.path.exists(saver_path): # if a pattern search save file exists
            with open(saver_path, 'rb') as f:
                self.saver = pickle.load(f)

            # self.saver['load'] = True # flags that a save file was loaded

            if self.saver['phase'] == 1:
                # If we had completed self.phase0, phase==1 load phase0 variables and run phase1
                # obj = self.saver['obj']
                # curfit = self.saver['fit']
                # best_tuners = self.saver['tuner']
                # mu = self.saver['mu']
                # iter = self.saver['iter']
                # self.saver['load'] = False
                self.phase1()


            elif self.saver['phase'] == 2:
                # If we had completed completed self.phase1, phase==2 load phase1 variables and run phase2
                # ...
                self.phase2()


                # ...
                self.phase1()


            else:
                # This should almost never be used because that means it was stopped at the beginning at is worth just restarting
                self.phase0()
                self.phase1()


            # print('Minimum fitness is {curfit}'.format(curfit=curfit))
            #
            # print('Optimal hyperparameters')
            # for ii in range(len(obj)):
            #     print('{ii} == {objs}'.format(ii=ii, objs=obj[0, ii]))

        else:
            # not saved, first run through?
            # self.saver['load'] = False
            # Begin random generation

            self.saver['savepath'] = saver_path
            self.saver['master_dict'] = master_dict
            self.saver['dict_template'] = dict_template
            self.phase0()
            self.phase1()

            with open(self.saver['savepath'], 'wb') as f:
                pickle.dump(self.saver, f)





        return self.saver['obj'], self.saver['tuners'], self.saver['master_dict']
        # return obj


    def save_checkpoint_results(self, dict_template, count, results, obj, tuner, best=True):
    # def save_checkpoint_results(self, dict_template, count, results):

        try:
            with open(self.checkpoint_filepath, 'rb') as f:
                cresults = pickle.load(f)

                bestcount = cresults[dict_template + '_bestcount']
        except:
            cresults = dict()
            bestcount = []

        cresults[dict_template + '{}'.format(count)] = results
        cresults[dict_template + '_obj_' + '{}'.format(count)] = obj
        cresults[dict_template + '_tuner_' + '{}'.format(count)] = tuner
        if best:
            bestcount.append(count)
            cresults[dict_template + '_bestcount'] = bestcount
        with open(self.checkpoint_filepath, 'wb') as f:
            pickle.dump(cresults, f)

    def build_null_labels(self, labs, proj, num_nulls):
        if labs.shape[1] > labs.shape[0]:
            labs = labs.T

        nullset = np.empty(labs.shape)
        null_arr = np.empty((labs.shape[0], labs.shape[1], num_nulls))

        if proj.shape[1] != labs.shape[1]:
            proj = proj.T
        for num in range(num_nulls):
            for ii in range(len(nullset)):
                ri = np.random.randint(0, len(proj))
                nullset[ii, :] = proj[ri, :]
            null_arr[:, :, num] = nullset

        return null_arr

    def gen_random_loss(self):
        return np.random.rand()

    def train_best_hps(self, tuners, objectives, train_epochs, num_nulls=50, predict=False, saver_path='untitled',
                       curdim=0, import_model=None
                       ):

        obj = self.enforce_bounds(objectives)
        obj = self.intcons(obj)
        obj = self.build_objdataset(obj)




        dimsout = self.output_dims[obj[0, 2]]
        if num_nulls > 0 and self.nn_architectures[obj[0, 6]] == 'ff':
            null_labels = self.build_null_labels(self.test_labels,
                                             self.projections[obj[0, 2]],
                                             num_nulls
                                             )
            resnull = np.empty((dimsout, train_epochs, num_nulls))
        elif num_nulls > 0 and self.nn_architectures[obj[0, 6]] == 'rnn':
            null_labels = self.build_null_labels(self.ordtest_labels,
                                             self.projections[obj[0, 2]],
                                             num_nulls
                                             )
            resnull = np.empty((dimsout, train_epochs, num_nulls))

        else:
            null_labels = None

        restr = np.empty((dimsout, train_epochs))
        reste = np.empty((dimsout, train_epochs))



        if os.path.exists(saver_path):
            with open(saver_path, 'rb') as f:
                saver = pickle.load(f)

            # restr = saver['restr']
            # reste = saver['reste']
            # resnull = saver['resnull']

        else:
            saver = dict()


        if self.nn_architectures[obj[0, 6]] == 'ff':

            for dim in range(curdim, dimsout):
                best_hps = tuners[dim].get_best_hyperparameters(num_trials=1)[0]
                if import_model is not None:
                    model = import_model[dim]
                else:
                    model = tuners[dim].hypermodel.build(best_hps)


                print('---------------------------------------------------------------------------------------------')
                print('Starting dimension {dim}'.format(dim=dim))
                print('---------------------------------------------------------------------------------------------')
                FF = Atom_FFNN(
                    data=self.train,
                    train_labels=self.train_labels,
                    test=self.test,
                    test_labels=self.test_labels,
                    nullset=self.test,
                    nullset_labels=null_labels,
                    model=model,
                    current_dim=dim,
                    save_model_path=self.savepath_model[dim],
                    optimize=False,
                    regression=True,
                    epochs=1,
                )
                FF.save_model()
                for ep in range(train_epochs):
                    print('Epoch: {ep}/{train_epochs}'.format(ep=ep, train_epochs=train_epochs))
                    FF = Atom_FFNN(
                        data=self.train,
                        train_labels=self.train_labels,
                        test=self.test,
                        test_labels=self.test_labels,
                        nullset=self.test,
                        nullset_labels=null_labels,
                        current_dim=dim,
                        model_path=self.savepath_model[dim],
                        save_model_path=self.savepath_model[dim],
                        optimize=False,
                        regression=True,
                        epochs=1,
                    )

                    history = FF.train()
                    trloss = history.history['loss']
                    restr[dim, ep] = trloss[0]
                    FF.save_model()
                    reste[dim, ep] = FF.test()

                    if num_nulls > 0:
                        for jj in range(num_nulls):
                            FF = Atom_FFNN(
                                data=self.train,
                                train_labels=self.train_labels,
                                test=self.test,
                                test_labels=self.test_labels,
                                nullset=self.test,
                                nullset_labels=null_labels[:, :, jj],
                                current_dim=dim,
                                model_path=self.savepath_model[dim],
                                save_model_path=self.savepath_model[dim],
                                optimize=False,
                                regression=True,
                                epochs=1,
                            )
                            resnull[dim, ep, jj] = FF.test_nullset()
                    keras.backend.clear_session()

                saver['restr'] = restr
                saver['reste'] = reste
                if 'resnull' in locals():
                    saver['resnull'] = resnull

                with open(saver_path, 'wb') as f:
                    pickle.dump(saver, f)

        else:
            for dim in range(dimsout):
                best_hps = tuners[dim].get_best_hyperparameters(num_trials=1)[0]
                if import_model is not None:
                    model = import_model[dim]
                else:
                    model = tuners[dim].hypermodel.build(best_hps)
                print('---------------------------------------------------------------------------------------------')
                print('Starting dimension {dim}'.format(dim=dim))
                print('---------------------------------------------------------------------------------------------')
                RNN = Atom_RNN(
                    data=self.ordtrain,
                    train_labels=self.ordtrain_labels,
                    test=self.ordtest,
                    test_labels=self.ordtest_labels,
                    nullset=self.ordtest,
                    nullset_labels=null_labels,
                    model=model,
                    steps=self.rnn_steps,
                    curdim=dim,
                    save_model_path=self.savepath_model[dim],
                    optimize=False,
                    regression=True,
                    epochs=1,
                )
                RNN.save_model()

                for ep in range(train_epochs):
                    print('Epoch: {ep}/{train_epochs}'.format(ep=ep, train_epochs=train_epochs))
                    RNN = Atom_RNN(
                        data=self.ordtrain,
                        train_labels=self.ordtrain_labels,
                        test=self.ordtest,
                        test_labels=self.ordtest_labels,
                        nullset = self.ordtest,
                        nullset_labels= null_labels,
                        steps = self.rnn_steps,
                        curdim=self.curout_dim,
                        model_path=self.savepath_model[dim],
                        save_model_path=self.savepath_model[dim],
                        optimize = False,
                        regression=True,
                        epochs = 1,
                    )

                    history = RNN.train()
                    trloss = history.history['loss']
                    restr[dim, ep] = trloss[0]
                    RNN.save_model()
                    reste[dim, ep] = RNN.test()
                    if num_nulls > 0:
                        for jj in range(num_nulls):
                            RNN = Atom_RNN(
                                data=self.ordtrain,
                                train_labels=self.ordtrain_labels,
                                test=self.ordtest,
                                test_labels=self.ordtest_labels,
                                nullset=self.ordtest,
                                nullset_labels=null_labels[:, :, jj],
                                steps=self.rnn_steps,
                                curdim=self.curout_dim,
                                model_path=self.savepath_model[dim],
                                save_model_path=self.savepath_model[dim],
                                optimize=False,
                                regression=True,
                                epochs=1,
                            )
                            resnull[dim, ep, jj] = RNN.test_nullset()
                    keras.backend.clear_session()
                saver['restr'] = restr
                saver['reste'] = reste
                if 'resnull' in locals():
                    saver['resnull'] = resnull

                with open(saver_path, 'wb') as f:
                    pickle.dump(saver, f)

        if predict:
            prediction = []; predlabels = []

            if self.nn_architectures[obj[0, 6]] == 'ff':
                for dim in range(dimsout):
                    FF = Atom_FFNN(
                        data=self.train,
                        train_labels=self.train_labels,
                        test=self.test,
                        test_labels=self.test_labels,
                        nullset=self.test,
                        nullset_labels=null_labels,
                        current_dim=dim,
                        model_path=self.savepath_model[dim],
                        save_model_path=self.savepath_model[dim],
                        optimize=False,
                        regression=True,
                        epochs=1,
                    )
                    prediction.append(FF.predict())
                predlabels.append(self.train_labels)

            else:
                for dim in range(dimsout):
                    RNN = Atom_RNN(
                        data=self.ordtrain,
                        train_labels=self.ordtrain_labels,
                        test=self.ordtest,
                        test_labels=self.ordtest_labels,
                        nullset=self.ordtest,
                        nullset_labels=null_labels,
                        steps=self.rnn_steps,
                        curdim=self.curout_dim,
                        model_path=self.savepath_model[dim],
                        save_model_path=self.savepath_model[dim],
                        optimize=False,
                        regression=True,
                        epochs=1,
                    )
                    prediction.append(RNN.predict())
                predlabels.append(self.ordtrain_labels)
        else:
            prediction = None
            predlabels= None




        if num_nulls > 0 and predict:
            return [restr, reste, resnull], [prediction, predlabels]

        elif predict:
            return [restr, reste], [prediction, predlabels]

        else:
            return [restr, reste]


    def predict(self, obj, models):

        self.build_objdataset(obj)
        dimsout = len(models)
        prediction = []
        predlabels = []


        if self.nn_architectures[obj[0, 6]] == 'ff':
            for dim in range(dimsout):
                FF = Atom_FFNN(
                    data=self.train,
                    train_labels=self.train_labels,
                    test=self.test,
                    test_labels=self.test_labels,
                    nullset=self.test,
                    # nullset_labels=null_labels,
                    current_dim=dim,
                    model_path=models[dim],
                    # save_model_path=self.savepath_model[dim],
                    optimize=False,
                    regression=True,
                    epochs=1,
                )
                prediction.append(FF.predict())
            predlabels.append(self.train_labels)

        else:
            for dim in range(dimsout):
                RNN = Atom_RNN(
                    data=self.ordtrain,
                    train_labels=self.ordtrain_labels,
                    test=self.ordtest,
                    test_labels=self.ordtest_labels,
                    nullset=self.ordtest,
                    # nullset_labels=null_labels,
                    steps=self.rnn_steps,
                    curdim=self.curout_dim,
                    model_path=models[dim],
                    # save_model_path=self.savepath_model[dim],
                    optimize=False,
                    regression=True,
                    epochs=1,
                )
                prediction.append(RNN.predict())
            predlabels.append(self.ordtrain_labels)

        return prediction, predlabels

    def parent_select(self, obj, fit, numparents, Y=2):
        # tournament selection parents
        # Y = 2 is most common and is the current default.
        # changing Y changes selection pressure
        parents = np.zeros([numparents, obj.shape[1]], dtype='int32')

        for ii in range(numparents):
            r = np.random.randint(0, self.npop, (Y,)) # random indices to pick competing parents
            randfits = []
            for jj in range(Y):
                randfits.append(fit[r[jj]]) # group parents we want to compare
            m = min(randfits)

            idx = np.where(fit==m)
            bo = np.issubdtype(type(idx), np.integer)
            while not bo:
                idx = idx[0]
                bo = np.issubdtype(type(idx), np.integer)
            parents[ii, :] = obj[idx, :]

        return parents

    def crossover(self, parents, mutation_rate):
        # one point crossover
        children = np.zeros(parents.shape, dtype='int32')
        assert children.shape[1] == self.nvar, "Children should have self.nvar genes"
        crosspoint = np.random.randint(0, self.nvar)
        if parents.shape[0] % 2 != 0:
            raise ValueError("numparents must be an even integer")
        shuffled = parents
        np.random.shuffle(shuffled) # shuffles parents along axis 0
        for ii in range(0, len(parents), 2):
            children[ii, :crosspoint] = shuffled[ii, :crosspoint]
            children[ii, crosspoint:] = shuffled[ii+1, crosspoint:]
            for jj in range(children.shape[1]):
                # mutation
                rmut = np.random.rand()
                if rmut < mutation_rate:
                    children[ii, jj] = np.random.randint(self.lb[jj], self.ub[jj])
            children[ii+1, :crosspoint] = shuffled[ii+1, :crosspoint]
            children[ii+1, crosspoint:] = shuffled[ii, crosspoint:]
            for jj in range(children.shape[1]):
                # mutation
                rmut = np.random.rand()
                if rmut < mutation_rate:
                    children[ii+1, jj] = np.random.randint(self.lb[jj], self.ub[jj])

        return children


    def reload_kt_checkpoints(self):
        for filename in os.listdir(self.kt_master_dir):
            name, file_extension = os.path.splitext(filename)
            if '.pkl' not in file_extension:
                start = name.find('[')
                end = name.find(']')
                key = name[start:end+2]
                self.kt_dir[key] = name

    def genetic_alg(self, numparents, npop, gasaver, save_every, mutation=0.2, stagnate=False, trainfor=0, n_nulls=0):
        self.npop = npop
        if os.path.exists(gasaver):
            with open(gasaver, 'rb') as f:
                saver = pickle.load(f)

            obj = saver['obj']
            fit = saver['fit']
            iter = saver['iter']
            lowestfit = saver['lowestfit']
            # if os.path.exists(self.kt_master_dir):
            #     self.reload_kt_checkpoints()


        else:
            saver = dict()
            if numparents > npop:
                raise ValueError("numparents must be less than npop")

            obj = np.zeros((self.npop, self.nvar), dtype='int32')
            fit = np.zeros(self.npop)
            for jj in range(self.npop):
                for ii in range(self.nvar):
                    # generates npop random solutions
                    obj[jj, ii] = np.random.randint(self.lb[ii], self.ub[ii])
            iter = 0
            lowestfit = np.inf
            saver['lfits'] = []
            saver['better_iters'] = []
            saver['results'] = []
            saver['mintuners'] = []

        while iter < self.maxiter:
            if stagnate: # reduce mutation rate over time?
                mutation_rate = mutation * (self.maxiter - iter)/self.maxiter
            else:
                mutation_rate = mutation

            poptuners = []
            obj = self.enforce_bounds(obj)
            obj = self.intcons(obj)
            for jj in range(self.npop):
                obj2d = obj[jj, :]
                obj2d = np.reshape(obj2d, (1, obj2d.shape[0]))
                obj2d = self.enforce_bounds(obj2d)
                obj2d = self.intcons(obj2d)
                obj2d= self.build_objdataset(obj2d)
                fit[jj], tuners = self.ff_fitness(
                    self.input_dims[obj[jj, 1]],
                    self.output_dims[obj[jj, 2]],
                    self.keyword_weigth_factor[obj[jj, 3]],
                    obj2d,
                    optimizer=self.optimizers[obj[jj, 5]],
                    nn_architecture=self.nn_architectures[obj[jj, 6]],
                    hyperoptimizer=self.hyperoptimizers[obj[jj, 7]]
                )
                poptuners.append(tuners)
            with open(self.kt_master_dir + '/fits.pkl', 'wb') as f:
                pickle.dump(self.kt_fits, f)

            with open(self.kt_master_dir + '/tuners.pkl', 'wb') as f:
                pickle.dump(self.trained_tuners, f)


            minfit = min(fit)
            if minfit < lowestfit:
                lowestfit = minfit
                m = np.where(fit == minfit)
                midx = m[0]
                minidx = midx[0]
                minobj = obj[minidx, :]
                minobj = np.reshape(minobj, (1, minobj.shape[0]))
                minobj = self.intcons(minobj)
                mintuners = poptuners[minidx]

                if trainfor > 0:
                    results = self.train_best_hps(mintuners, minobj, trainfor, n_nulls,
                                                  saver_path=os.path.join(up_dir,'Learn_Master/GA_files/train_saver.pkl'))
                    allresults = saver['results']
                    allresults.append(results)
                    saver['lfits'].append(lowestfit)
                    saver['better_iters'].append(iter)
                    saver['results'] = allresults
                    saver['lowestfit'] = lowestfit
                    saver['mintuners'].append(mintuners)

            if iter % save_every == 0:
                saver['fit'] = fit
                saver['obj'] = obj
                saver['iter'] = iter
                saver['tuners'] = poptuners
                with open(gasaver, 'wb') as f:
                    pickle.dump(saver, f)


            iter += 1
            parents = self.parent_select(obj, fit, numparents, Y=2)
            children = self.crossover(parents, mutation_rate)

            obj = np.concatenate((parents, children), axis=0)
            np.random.shuffle(obj)

        with open(gasaver, 'wb') as f:
            pickle.dump(saver, f)

    # def timer_interrupt(self):
    #     import threading
    #
    #
    #     hours = 7.9
    #
    #     runtime = float(hours*3600)
    #     t = threading.Timer(runtime, self.save_checkpoint_results(dict_template, count, results, obj, tuner, model))



if __name__ == '__main__':


    paras_path = os.path.join(up_dir,'Misc_Data/raw_ricoparas.pkl')
    with open(paras_path, 'rb') as f:
        paras = pickle.load(f)

    doc_dict_path = os.path.join(up_dir,'Misc_Data/doc_dict.pkl')
    with open(doc_dict_path, 'rb') as f:
        doc_dict = pickle.load(f)

    encoded_data_paths = [
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/encoded_data_01.pkl'),
        os.path.join(up_dir,'Word_Embeddings/Scientific_Articles/ricocorpus_sciart_encoded.pkl')
    ]
    encoded_data = []
    for en in encoded_data_paths:
        with open(en, 'rb') as f:
            encoded_data.append(pickle.load(f))

    vocab_paths = [
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/ranked_vocab.pkl'),
        os.path.join(up_dir,'Word_Embeddings/Scientific_Articles/sciart_vocab.pkl')
    ]
    vocab = []
    for v in vocab_paths:
        with open(v, 'rb') as f:
            vocab.append(pickle.load(f))

    input_dims = [2, 3, 5, 10, 20, 30, 40, 50, 2]
    # input_dims = [2]
    output_dims = [2, 3, 4] # greatly increases computation time

    model_paths = []
    for ii in range(output_dims[-1]):
        if len(range(output_dims[-1])) >= 100:
            model_paths.append(os.path.join(up_dir, 'Learn_Master/optimized_models/model_00{ii}'.format(ii=ii)))
        else:
            model_paths.append(os.path.join(up_dir, 'Learn_Master/optimized_models/model_0{ii}'.format(ii=ii)))

    def fibonacci(n):
        fib = [1, 1]
        for ii in range(1, n):
            fib.append(fib[ii-1] + fib[ii])

        fib.remove(1) # removes first 1
        return fib

    weighting_factor = fibonacci(10)
    # weighting_factor = [1]
    optimizers = ['sgd', 'adam', 'adagrad', 'rmsprop', 'adadelta', 'adamax']

    label_paths = [
        os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_2Dims.pkl'),
        os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_3dims.pkl'),
        os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_4dims.pkl'),
        # os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_5dims.pkl'),
        # os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_6dims.pkl'),
        # os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_7dims.pkl'),
        # os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_8dims.pkl'),
        # os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_9dims.pkl'),
        # os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_10Dims.pkl')
                   ]
    labels = []
    for la in label_paths:
        with open(la, 'rb') as f:
            labs = pickle.load(f)

            labs = np.asarray(labs)
            labels.append(labs)

    proj_paths = [
        os.path.join(up_dir,'Misc_Data/Projection_2dims.pkl'),
        os.path.join(up_dir,'Misc_Data/Projection_3dims.pkl'),
        os.path.join(up_dir,'Misc_Data/Projection_4dims.pkl'),
        # os.path.join(up_dir,'Misc_Data/Projection_5dims.pkl'),
        # os.path.join(up_dir,'Misc_Data/Projection_6dims.pkl'),
        # os.path.join(up_dir,'Misc_Data/Projection_7dims.pkl'),
        # os.path.join(up_dir,'Misc_Data/Projection_8dims.pkl'),
        # os.path.join(up_dir,'Misc_Data/Projection_9dims.pkl'),
        # os.path.join(up_dir,'Misc_Data/Projection_10dims.pkl'),
    ]
    projections = []
    for pr in proj_paths:
        with open(pr, 'rb') as f:
            projections.append(pickle.load(f))




    we_model_paths = [
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_2dims_v02'),
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_3dims_v02'),
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_5dims_v02'),
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_10dims_v02'),
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_20dims_v02'),
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_30dims_v02'),
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_40dims_v02'),
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_50dims_v02'),
        os.path.join(up_dir,'Word_Embeddings/Scientific_Articles/sciart_model')
    ]






    # atom_embed_method = ['sum_atoms', 'avg_atoms', 'ndelta', 'pvdm']
    atom_embed_method = ['sum_atoms', 'avg_atoms', 'ndelta']
    # atom_embed_method = ['sum_atoms']
    # for output in output_dims:
    #     for curdim in range(output):
    #         Learn_Master(
    #
    #         )

    hyperoptimizers = ['random', 'bayes', 'hyper']

    ints = [0, 1, 2, 3, 4, 5, 6, 7]

    nn_arch = ['ff', 'rnn']
    # nn_arch = ['ff']
    curdim = 0

    run_name = 'ga_tu25_pop20_tr25_stag_03'
    # run_name = 'test'
    saverpath = '{}.pkl'.format(run_name)

    # kt_master_dir = '//ocean/projects/mss140007p/danewebb/kt_checkpoints'
    kt_master_dir = os.path.join(up_dir, 'Learn_Master/kt_checkpoints')
    # kt_master_dir = r'D:\kt_checkpoints'

    fit_savepath = os.path.join(up_dir, 'Learn_Master/checkpoints/fitness_{run_name}.pkl'.format(run_name=run_name))
    # run_name = 'test'
    checkpoint_filepath = os.path.join(up_dir,'Learn_Master/checkpoints/cresults_{run_name}.pkl'.format(run_name=run_name))
    # checkpoint_filepath = os.path.join(up_dir,'Learn_Master/checkpoints/test.pkl')
    LM = Learn_Master(
        doc_dict,
        vocab,
        encoded_data,
        labels,
        input_dims,
        we_model_paths,
        output_dims,
        weighting_factor,
        optimizers,
        atom_embed_method,
        nn_arch,
        hyperoptimizers,
        ints,
        projections,
        # curout_dim=curdim,
        epochs = 100,
        fitness_epochs= 100,
        mu0=3,
        alpha=1,
        delta=1,
        maxiter=50,
        savepath_model=model_paths,
        rnn_steps=3,
        kt_master_dir=kt_master_dir,
        checkpoint_filepath=checkpoint_filepath,
        raw_paras=paras,
        fitness_savepath=fit_savepath,
    )


### GA script
    numparents = 10
    npop = 20
    mutate_rate = 0.3
    ga_run_name = 'ga_tu25_pop20_tr25_stag_03'


    gasaver = os.path.join(up_dir,'Learn_Master/GA_files/{}.pkl'.format(ga_run_name))

    LM.genetic_alg(numparents, npop, gasaver, 1, mutation=mutate_rate, stagnate=True, trainfor=25, n_nulls=0)

    # with open(os.path.join(up_dir,'Learn_Master/optimized_models/ga_results_{run_name}.pkl'.format(run_name=run_name)), 'wb') as f:
    #     pickle.dump(optimization_results, f)
    #
    #
    # with open(os.path.join(up_dir,'Learn_Master/best/ga_objectives_{run_name}.pkl'.format(run_name=run_name)), 'wb') as f:
    #     pickle.dump(objectives, f)
    # with open(os.path.join(up_dir,'Learn_Master/best/ga_tuner_{run_name}.pkl'.format(run_name=run_name)), 'wb') as f:
    #     pickle.dump(tuner, f)


### GA Train and predict
    # pred_name = 'ga_tu25_pop20_tr25_stag_pred'
    #
    #
    #
    # with open(os.path.join(up_dir, 'Learn_Master/GA_files/ga_tu25_pop20_tr25_stag.pkl'), 'rb') as f:
    #     graph_dict = pickle.load(f)
    #
    # kerasmodel = keras.models.load_model(os.path.join(up_dir, 'Shuffled_Data_1/Rico-Corpus/model_10000ep_10dims/BOWavg_rico/Output_10Dims/model50_10dims_03'))
    #
    # # bestcount, _ = find_mins(ftr)
    #
    # pred_res = []
    # train_res = []
    #
    # listoftuners = graph_dict['mintuners']
    # tuners = listoftuners[0]
    #
    # best_hps = tuners[0].get_best_hyperparameters(num_trials=1)[0]
    #
    # model = tuners[0].hypermodel.build(best_hps)
    #
    # obj = [1, 7, 2, 1, 3, 5, 0, 1]
    # obj = np.asarray(obj)
    # obj = np.reshape(obj, (1, obj.shape[0]))
    #
    #
    #
    # epochs = 500
    #
    # # for jj, idx in enumerate(bestcount):
    # tres, pres = LM.train_best_hps(tuners, obj, epochs, num_nulls=25, predict=True,
    #                                saver_path=os.path.join(up_dir,'Learn_Master/GA_files/saver.pkl'),
    #                                curdim=2,
    #                                )
    # # print('Complete round {} of {}'.format(jj, len(bestcount)))
    # pred_res.append(pres)
    # train_res.append(tres)
    # with open(os.path.join(up_dir, 'Learn_Master/predictions/chk/chkprediction_{name}.pkl'.format(name=pred_name)), 'wb') as f:
    #     pickle.dump(pred_res, f)
    # with open(os.path.join(up_dir, 'Learn_Master/predictions/chk/chkresults_{name}.pkl'.format(name=pred_name)), 'wb') as f:
    #     pickle.dump(train_res, f)
    #
    # with open(os.path.join(up_dir, 'Learn_Master/predictions/prediction_{}.pkl'.format(pred_name)), 'wb') as f:
    #     pickle.dump(pred_res, f)
    #
    # with open(os.path.join(up_dir,'Learn_Master/predictions/results_{}'.format(pred_name)), 'wb') as f:
    #     pickle.dump(train_res, f)
    #


### GA Train and Predict kt_checkpoints

    # with open(os.path.join(up_dir, 'fits.pkl'), 'rb') as f:
    #     fits =  pickle.load(f)
    #
    # with open(os.path.join(up_dir, 'tuners.pkl'), 'rb') as f:
    #     tuners =  pickle.load(f)
    #
    # minfit = 1
    # for key, val in fits.items():
    #     if val < minfit:
    #         minfit = val
    #         keywant = key
    #
    # strobj = []
    # for ele in keywant:
    #     if ele == '[' or ele == ']':
    #         pass
    #     else:
    #         strobj.append(ele)
    #
    # strobj = ''.join(strobj)
    #
    # obj = np.fromstring(strobj, dtype=int, sep=' ')
    # obj = np.reshape(obj, (1, obj.shape[0]))
    #
    #
    #
    # pred, predlabels = LM.train_best_hps(tuners[keywant], obj, 500, num_nulls=0, predict=True)
    # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\pred.pkl', 'wb') as f:
    #     pickle.dump(pred, f)
    #
    # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\predlabels.pkl', 'wb') as f:
    #     pickle.dump(predlabels, f)

        # def train_best_hps(self, tuners, objectives, train_epochs, num_nulls=50, predict=False, saver_path='untitled',
        #                    curdim=0, import_model=None
        #                    ):





### PS script
#     try:
#         with open(os.path.join(up_dir,'Learn_Master/optimized_models/results_{run_name}.pkl'.format(run_name=run_name)), 'rb') as f:
#             optimization_results = pickle.load(f)
#     except:
#         optimization_results = dict()
#     results = dict()
#
#     dict_template = 'improved_meta_set'
#     objectives, tuner, optimization_results['optimizing'] = LM.PS_integer(train_for=1, dict_template=dict_template, master_dict=results, n_nulls=0, saver_path=saverpath)
#
#     with open(os.path.join(up_dir,'Learn_Master/optimized_models/results_{run_name}.pkl'.format(run_name=run_name)), 'wb') as f:
#         pickle.dump(optimization_results, f)
#
#
#     with open(os.path.join(up_dir,'Learn_Master/best/objectives_{run_name}.pkl'.format(run_name=run_name)), 'wb') as f:
#         pickle.dump(objectives, f)
#     with open(os.path.join(up_dir,'Learn_Master/best/tuner_{run_name}.pkl'.format(run_name=run_name)), 'wb') as f:
#         pickle.dump(tuner, f)
    # with open(os.path.join(up_dir,'Learn_Master/best/model.pkl'), 'wb') as f:
    #     pickle.dump(model, f)
    # for ii, mod in enumerate(model):
    #     keras.models.save_model(model, 'Learn_Master/best/model_{run_name}_{ii}'.format(run_name=run_name, ii=ii)) # don't want to save list of models





### prediction script
    # from Lib.Misc_Data.Pre_Process import breakup
    # pred_name = 'ga_tu25_pop20_tr25_stag_pred'
    # num = 84
    #
    # dim = 0
    # with open(os.path.join(up_dir, 'Learn_Master/checkpoints/cresults_{}.pkl'.format(pred_name)), 'rb') as f:
    #     graph_dict = pickle.load(f)
    #
    #
    # bbestcount = graph_dict['improved_metaset_actual_bestcount']
    # bestcount = bbestcount[1:]
    # train, test, ftr, fte = breakup(graph_dict, num, dim)
    # # bestcount, _ = find_mins(ftr)
    # tuners = []
    # objectives = []
    # train_res = []
    # pred_res = []
    # act = []
    # for idx in bestcount:
    #     key = 'improved_meta_set_tuner_{}'.format(idx)
    #     tuners.append(graph_dict[key])
    #     key = 'improved_meta_set_obj_{}'.format(idx)
    #     objectives.append(graph_dict[key])
    #
    #
    #
    # epochs = 500
    #
    # # for jj, idx in enumerate(bestcount):
    # tres, pres = LM.train_best_hps(tuners[-1], objectives[-1], epochs, num_nulls=5, predict=True)
    # # print('Complete round {} of {}'.format(jj, len(bestcount)))
    # pred_res.append(pres)
    # train_res.append(tres)
    # with open(os.path.join(up_dir, 'Learn_Master/predictions/chk/chkprediction_{name}_2.pkl'.format(name=pred_name)), 'wb') as f:
    #     pickle.dump(pred_res, f)
    # with open(os.path.join(up_dir, 'Learn_Master/predictions/chk/chkresults_{name}_2.pkl'.format(name=pred_name)), 'wb') as f:
    #     pickle.dump(train_res, f)
    #
    # with open(os.path.join(up_dir, 'Learn_Master/predictions/prediction_{}.pkl'.format(pred_name)), 'wb') as f:
    #     pickle.dump(pred_res, f)
    #
    # with open(os.path.join(up_dir,'Learn_Master/predictions/results_{}'.format(pred_name)), 'wb') as f:
    #     pickle.dump(train_res, f)


### GA prediction script
    # pred_name = 'ga_tu25_pop20_tr25_stag_pred'
    #
    # optimized_models = [os.path.join(up_dir,'Learn_Master/optimized_models/model_00'),
    #                     os.path.join(up_dir,'Learn_Master/optimized_models/model_01')]
    # obj = [1, 7, 2, 1, 3, 5, 0, 1]
    # obj = np.asarray(obj)
    # obj = np.reshape(obj, (1, obj.shape[0]))
    #
    # prediction, predlabels = LM.predict(obj, optimized_models)
    #
    #
    # with open( os.path.join(up_dir, 'Learn_Master/predictions/pred_{}.pkl'.format(pred_name)), 'wb') as f:
    #     pickle.dump(prediction, f)
    #
    # with open(os.path.join(up_dir, 'Learn_Master/predictions/predlabels_{}.pkl'.format(pred_name)), 'wb') as f:
    #     pickle.dump(predlabels, f)


