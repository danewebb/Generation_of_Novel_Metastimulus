import sys
sys.path.append("..") # Adds higher directory to python modules path.
import os
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
up_dir = os.path.join(script_dir, '..')

# from atom_tag_pairing import Atom_Tag_Pairing


from Lib.Atom_Embeddings.atom_embedding import Atom_Embedder
from Lib.Metastimuli_Learn.Atom_FFNN.Atom_FFNN import Atom_FFNN
from Lib.Metastimuli_Learn.Atom_RNN.Atom_RNN import Atom_RNN
from Lib.Word_Embeddings.Weighting_Keywords import Weighting_Keyword
# fromword_embedding_ricocorpus import word_embedding_ricocorpus
# from word_embedding import Sciart_Word_Embedding
from Lib.Shuffled_Data_1.Randomizer import Shuffler

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

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Learn_Master():
    """

    """
    def __init__(self, dataset_dict, vocab, ordered_encdata, ordered_labels, input_dims, we_models, output_dims, keyword_weight_factor,
                 optimizers, atom_methods, nn_architectures, hyperoptimizers, ints, projections=None,
                 savepath_model = None, kt_directory='untitled',
                 NN_prebuilt_model=None, NNclasse=None,
                 WE_classes=None, vocab_save_path='', data_save_path='', checkpoint_filepath='untitled_checkpoints',
                 projection=None, adjacency=None, curout_dim=0, raw_paras=None,
                 epochs=1, fitness_epochs=10, rnn_steps=3,

                 maxiter=100, mindiff=1e-6, mu0=4, alpha=1e-1, delta=4,
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

        self.kt_dir= kt_directory

        self.bestcount = []

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
            kpen = 0.01 * wkey
            kpen = kpen + 1
        else:
            kpen = 1

        if out > 2:
            open = 0.05*out
            open = open + 1
        else:
            open = 1

        penloss = loss*kpen*open
        return penloss



    def ff_fitness(self, input_dim, out_dims, kweight, optimizer='sgd', nn_architecture='ff', hyperoptimizer='random'):
        total_loss = []
        all_models = []
        all_tuners = []

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
        penloss = self.penalties(avgloss, kweight, out_dims)


        return penloss, all_models, all_tuners



    def make_mesh(self, x):
        jj = 0
        mesh = np.zeros((2*self.nvar, self.nvar))
        gps = np.zeros((2 * self.nvar, self.nvar))

        for ii in range(mesh.shape[1]):
            if ii <= self.nvar:
                mesh[ii, jj] = self.mu
                jj += 1
                if ii == self.nvar:
                    jj = 0
            else:
                mesh[ii, jj] = -self.mu
                jj += 1

        for ii in range(mesh.shape[1]):
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


    def PS_integer(self, train_for=0, dict_template='', master_dict=None, n_nulls=50):
        if master_dict is None:
            master_dict = dict()
        count = 0
        obj = np.zeros((1, self.nvar), dtype='int32')
        for ii in range(self.nvar):
            obj[0, ii] = np.random.randint(self.lb[ii], self.ub[ii])

        dic = dict()
        dic['f'] = [1, 2, 3, 4]
        # self.save_checkpoint_results(dict_template, 5, dic)
        objold = np.zeros((1, self.nvar), dtype='int32')
        objnew = np.zeros((1, self.nvar), dtype='int32')
        mu = self.mu0

        oldfit = np.inf
        newfit = 0

        obj = self.build_objdataset(obj)

        curfit, best_models, best_tuners = self.ff_fitness(
            self.input_dims[obj[0, 1]],
            self.output_dims[obj[0, 2]],
            self.keyword_weigth_factor[obj[0, 3]],
            optimizer=self.optimizers[obj[0, 5]],
            nn_architecture = self.nn_architectures[obj[0, 6]],
            hyperoptimizer=self.hyperoptimizers[obj[0, 7]]
        )

        if train_for > 0:
            count += 1
            results = self.train_best_hps(best_tuners, obj, train_for, num_nulls=n_nulls)
            self.save_checkpoint_results(dict_template, count, results, obj, best_tuners, best=False)
        diff = np.inf
        iter = 0

        while iter < self.maxiter:

            import time
            #  and diff > self.mindiff:
            if iter % 100:
                t = time.localtime()
                now = time.strftime("%H:%M:%S", t)
                print('-----------------------------------------------------------------------------------------------')
                print('-----------------------------------------------------------------------------------------------')
                print('-----------------------------------------------------------------------------------------------')
                print('Begin iteration {i} at {n}'.format(i = iter, n = now))
                print('-----------------------------------------------------------------------------------------------')
                print('-----------------------------------------------------------------------------------------------')
                print('-----------------------------------------------------------------------------------------------')
            changeflag = 0
            mesh = self.make_mesh(obj)
            mesh = self.enforce_bounds(mesh)
            mesh = self.intcons(mesh)

            # check if any of the exploratory points have lower fitness
            for ii in range(1, self.nvar):
                mesh[ii, :] = self.build_objdataset(mesh[ii, :])
                print(mesh[ii, :])
                meshfit, meshmods, tuners = self.ff_fitness(
                    self.input_dims[mesh[ii, 1]],
                    self.output_dims[mesh[ii, 2]],
                    self.keyword_weigth_factor[mesh[ii, 3]],
                    optimizer=self.optimizers[mesh[ii, 5]],
                    nn_architecture=self.nn_architectures[mesh[ii, 6]],
                    hyperoptimizer=self.hyperoptimizers[mesh[ii, 7]]
                )
                if train_for > 0:
                    count += 1
                    results = self.train_best_hps(tuners, mesh[ii, :], train_for, num_nulls=n_nulls)
                    self.save_checkpoint_results(dict_template, count, results, mesh[ii, :], tuners, best=False)

                if meshfit < curfit:
                    curfit = meshfit
                    objnew = mesh[ii, :]
                    objnew = np.reshape(objnew, (1,len(objnew)))
                    best_model = meshmods
                    best_tuner = tuners
                    changeflag = 1
                    start = time.time()
                    if train_for > 0:

                        results = self.train_best_hps(best_tuner, objnew, train_for, num_nulls=n_nulls)
                        master_dict[dict_template+'{count}'.format(count=count)] = results
                        count += 1

                        self.save_checkpoint_results(dict_template, count, results, objnew, best_tuner, best=True)

            if changeflag == 1:
                changeflag = 0
                mu = self.mu0
                while newfit < curfit:
                    objold = obj
                    oldfit = curfit
                    obj = objnew
                    objnew = objold + self.alpha*(obj - objold)
                    objnew = self.enforce_bounds(objnew)
                    objnew = self.intcons(objnew)
                    objnew = self.build_objdataset(objnew)

                    newfit, newmodels, new_tuners = self.ff_fitness(
                        self.input_dims[objnew[0, 1]],
                        self.output_dims[objnew[0, 2]],
                        self.keyword_weigth_factor[objnew[0, 3]],
                        optimizer=self.optimizers[objnew[0, 5]],
                        nn_architecture=self.nn_architectures[objnew[0, 6]],
                        hyperoptimizer=self.hyperoptimizers[objnew[0, 7]]
                    )

                    if train_for > 0:
                        results = self.train_best_hps(new_tuners, objnew, train_for, num_nulls=n_nulls)
                        master_dict[dict_template+'{count}'.format(count=count)] = results
                        count += 1

                        self.save_checkpoint_results(dict_template, count, results, objnew, new_tuners, best=False)


                    mesh = self.make_mesh(objnew)
                    mesh = self.enforce_bounds(mesh)
                    mesh = self.intcons(mesh)

                    for ii in range(1, self.nvar):
                        self.build_objdataset(mesh[ii, :])

                        meshfit, meshmod, tuner = self.ff_fitness(

                            self.input_dims[mesh[ii, 1]],
                            self.output_dims[mesh[ii, 2]],
                            self.keyword_weigth_factor[mesh[ii, 3]],
                            optimizer=self.optimizers[mesh[ii, 5]],
                            nn_architecture=self.nn_architectures[mesh[ii, 6]],
                            hyperoptimizer=self.hyperoptimizers[mesh[ii, 7]]
                        )
                        if meshfit < curfit:
                            oldfit = curfit
                            curfit = meshfit
                            objnew = mesh[ii, :]
                            best_model = meshmod
                            best_tuner = tuner
                            changeflag = 1

                        if train_for > 0:
                            count += 1
                            results = self.train_best_hps(tuner, mesh[ii, :], train_for, num_nulls=n_nulls)
                            self.save_checkpoint_results(dict_template, count, results, mesh[ii, :], tuner, best=False)

                    if changeflag == 1:
                        changeflag = 0
                        if train_for > 0:
                            end = time.time()
                            results = self.train_best_hps(best_tuner, objnew, train_for, num_nulls=n_nulls)
                            master_dict[dict_template + '{count}'.format(count=count)] = results
                            count += 1
                            self.save_checkpoint_results(dict_template, count, results, objnew, best_tuner, best=True)

                            print('New center took {time} hours'.format(time = (end-start)/3600))

            else:
                mu = mu - self.delta
                iter += 1

            print('Minimum fitness is {curfit}'.format(curfit=curfit))

        print('Optimal hyperparameters')
        for ii in range(len(obj)):
            print('{ii} == {objs}'.format(ii=ii, objs=obj[0, ii]))


        return obj, best_tuner, best_model, master_dict


    def save_checkpoint_results(self, dict_template, count, results, obj, tuner, best=True):
    # def save_checkpoint_results(self, dict_template, count, results):

        try:
            with open(self.checkpoint_filepath, 'rb') as f:
                cresults = pickle.load(f)
        except:
            cresults = dict()

        cresults[dict_template + '{}'.format(count)] = results
        cresults[dict_template + '_obj_' + '{}'.format(count)] = obj
        cresults[dict_template + '_tuner_' + '{}'.format(count)] = tuner
        if best:
            self.bestcount.append(count)
            cresults[dict_template + '_bestcount'] = self.bestcount
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



    def train_best_hps(self, tuners, objectives, train_epochs, num_nulls=50, predict=False):



        obj = self.build_objdataset(objectives)

        dimsout = self.output_dims[obj[0, 2]]
        if num_nulls != 0 and self.nn_architectures[obj[0, 6]] == 'ff':
            null_labels = self.build_null_labels(self.test_labels,
                                             self.projections[obj[0, 2]],
                                             num_nulls
                                             )
            resnull = np.empty((dimsout, train_epochs, num_nulls))
        elif num_nulls != 0 and self.nn_architectures[obj[0, 6]] == 'rnn':
            null_labels = self.build_null_labels(self.ordtest_labels,
                                             self.projections[obj[0, 2]],
                                             num_nulls
                                             )
            resnull = np.empty((dimsout, train_epochs, num_nulls))

        else:
            null_labels = None

        restr = np.empty((dimsout, train_epochs))
        reste = np.empty((dimsout, train_epochs))

        if obj[0, 6] == 0:

            for dim in range(dimsout):
                best_hps = tuners[dim].get_best_hyperparameters(num_trials=1)[0]
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

        else:
            for dim in range(dimsout):
                best_hps = tuners[dim].get_best_hyperparameters(num_trials=1)[0]
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

        if predict:
            if obj[0, 6] == 0:
                prediction = FF.predict()
                predlabels = self.train_labels

            else:
                prediction = RNN.predict()
                predlabels = self.ordtrain_labels
        else:
            prediction = None
            predlabels= None




        if num_nulls > 0 and predict:
            return [restr, reste, resnull], [prediction, predlabels]

        elif predict:
            return [restr, reste], [prediction, predlabels]

        else:
            return [restr, reste]


    # def timer_interrupt(self):
    #     import threading
    #     import time
    #
    #     hours = 7.9
    #
    #     runtime = float(hours*3600)
    #     t = threading.Timer(runtime, self.save_checkpoint_results(self.dict_template, self.count, self.results, self.obj, self.tuner, self.model))



if __name__ == '__main__':


    paras_path = os.path.join(up_dir,'Misc_Data/raw_ricoparas.pkl')
    with open(paras_path, 'rb') as f:
        paras = pickle.load(f)

    doc_dict_path = os.path.join(up_dir,'Misc_Data/doc_dict.pkl')
    with open(doc_dict_path, 'rb') as f:
        doc_dict = pickle.load(f)

    encoded_data_paths = [
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/encoded_data_01.pkl'),
        os.path.join(up_dir,'Word_Embeddings/Scientific-Articles/ricocorpus_sciart_encoded.pkl')
    ]
    encoded_data = []
    for en in encoded_data_paths:
        with open(en, 'rb') as f:
            encoded_data.append(pickle.load(f))

    vocab_paths = [
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/ranked_vocab.pkl'),
        os.path.join(up_dir,'Word_Embeddings/Scientific-Articles/sciart_vocab.pkl')
    ]
    vocab = []
    for v in vocab_paths:
        with open(v, 'rb') as f:
            vocab.append(pickle.load(f))

    input_dims = [2, 3, 5, 10, 20, 30, 40, 50, 2]
    output_dims = [2, 3, 4, 5] # greatly increases computation time

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

    weighting_factor = fibonacci(20)

    optimizers = ['sgd', 'adam', 'adagrad', 'rmsprop', 'adadelta', 'adamax']

    label_paths = [
        os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_2Dims.pkl'),
        os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_3dims.pkl'),
        os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_4dims.pkl'),
        os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_5dims.pkl'),
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
        os.path.join(up_dir,'Misc_Data/Projection_5dims.pkl'),
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
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_2dims'),
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_3dims'),
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_5dims'),
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_10dims'),
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_20dims'),
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_30dims'),
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_40dims'),
        os.path.join(up_dir,'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_50dims'),
        os.path.join(up_dir,'Word_Embeddings/Scientific-Articles/sciart_model')
    ]






    atom_embed_method = ['sum_atoms', 'avg_atoms', 'ndelta', 'pvdm']


    # for output in output_dims:
    #     for curdim in range(output):
    #         Learn_Master(
    #
    #         )

    hyperoptimizers = ['random', 'bayes', 'hyper']

    ints = [0, 1, 2, 3, 4, 5, 6, 7]

    nn_arch = ['ff', 'rnn']
    curdim = 0

    checkpoint_filepath = 'Learn_Master/checkpoints/cresults_s05ep_pen01.pkl'
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
        epochs = 5,
        mu0=4,
        alpha=1,
        delta=1,
        maxiter=5,
        savepath_model=model_paths,
        rnn_steps=3,
        kt_directory='test1',
        checkpoint_filepath=checkpoint_filepath,
        raw_paras=paras,
    )
# PS script
#     try:
#         with open(os.path.join(up_dir,'Learn_Master/optimized_models/results_s05ep_pen01.pkl'), 'rb') as f:
#             optimization_results = pickle.load(f)
#     except:
#         optimization_results = dict()
#     results = dict()
#
#     dict_template = 'improved_meta_set'
#     objectives, tuner, model, optimization_results['optimizing'] = LM.PS_integer(train_for=2, dict_template=dict_template, master_dict=results, n_nulls=0)
#
#     with open('Learn_Master/optimized_models/results_s05ep_pen01.pkl', 'wb') as f:
#         pickle.dump(optimization_results, f)
#
#
#     with open('Learn_Master/best/objectives_s05ep_pen01.pkl', 'wb') as f:
#         pickle.dump(objectives, f)
#     with open('Learn_Master/best/tuner_s05ep_pen01.pkl', 'wb') as f:
#         pickle.dump(tuner, f)
    # with open(os.path.join(up_dir,'Learn_Master/best/model.pkl'), 'wb') as f:
    #     pickle.dump(model, f)
    # keras.models.save_model(model, 'Learn_Master/best/model_s05ep_pen01') # don't want to save list of models

# prediction script
    num = 72
    with open('../Learn_Master/checkpoints/cresults_s05ep_pen01.pkl', 'rb') as f:
        graph_dict = pickle.load(f)
    tuners = []
    objectives = []
    train_res = []
    pred_res = []
    act = []
    for ii in range(1, num+1):
        key = 'improved_meta_set_tuner_{}'.format(ii)
        tuners.append(graph_dict[key])
        key = 'improved_meta_set_obj_{}'.format(ii)
        objectives.append(graph_dict[key])

    epochs = 50
    for jj in range(0, num-1):
        _, pres = LM.train_best_hps(tuners[jj], objectives[jj], epochs, num_nulls=0, predict=True)

        pred_res.append(pres)
        obj = objectives[jj]
        if obj[0, 6] == 0:
            act.append(LM.train)
        else:
            act.append(LM.ordtrain)
    with open('../Learn_Master/prediction.pkl', 'wb') as f:
        pickle.dump(pred_res, f)
    with open('../Learn_Master/actual.pkl', 'wb') as f:
        pickle.dump(act, f)

