from atom_tag_pairing import Atom_Tag_Pairing
from atom_embedding import Atom_Embedder
from Atom_FFNN import Atom_FFNN
from Atom_RNN import Atom_RNN
# from word_embedding_ricocorpus import word_embedding_ricocorpus
# from word_embedding import Sciart_Word_Embedding
from Randomizer import Shuffler

from pathlib import Path
from Process_sciart import Process_Sciart
from Label_Text_Builder import Label_Text_Builder
from Processing import Tex_Processing

# PIMS filter import
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow import keras
import keras.layers as layers

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Learn_Master():
    """

    """
    def __init__(self, dataset_dict, vocab, ordered_encdata, ordered_labels, ints, input_dims, we_models, output_dims, keyword_weight_factor,
                 optimizers, atom_methods, nn_architectures, hyperoptimizers,
                 NN_prebuilt_model=None, NNclasse=None,
                 WE_classes=None, vocab_save_path='', data_save_path='',
                 projection=None, adjacency=None, curout_dim=0,

                 nvar=10, maxiter=100, mindiff=1e-6, mu0=4, alpha=1e-1, delta=4,
                 ):
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


        self.we_models = we_models


        self.mindiff = mindiff
        self.maxiter = maxiter
        self.mu0= mu0
        self.mu = mu0

        # need to define later
        self.list_train_paths = None
        self.list_train_label_paths = None
        self.list_test_paths = None
        self.list_test_label_paths = None

        self.ordtrain = None; self.ordtrain_labels = None; self.ordtest = None; self.ordtest_labels = None
        self.train = None; self.train_labels = None; self.test = None; self.test_labels = None

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

        from Weighting_Keywords import Weighting_Keyword
        if objectives.ndim == 1:
            obj = objectives.astype(int)
            obj = np.reshape(obj, (1, len(obj)))
        else:
            obj = objectives

        if obj[0, 1] == 4:
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

        encoded, nlabels = self.eliminate_empties(enc, labels)

        key_weight = self.keyword_weigth_factor[obj[0, 3]]

        atom_method = self.atom_methods[obj[0, 4]]

        atom_vecs = self.atom_embedding(atom_method, we_model, voc, key_weight, encoded)
        self.split_shuff(atom_vecs, nlabels, input_dimension)

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

        from atom_embedding import Atom_Embedder
        from Weighting_Keywords import Weighting_Keyword

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

    #     elif atom_embed_method == 'SIF'

        # with open(Path(f'C:\\Users\\liqui\\PycharmProjects\\Generation_of_Novel_Metastimulus\\Lib\\Atom-Embeddings\\'
        #                f'embeddings\\{we_embed}_w{self.keyword_weigth_factor}_in{atom_vecs[0].shape[0]}_{atom_embed_method}.pkl'),
        #           'wb') as f:
        #     pickle.dump(atom_vecs, f)

        return atom_vecs


    def split_shuff(self, atom_vecs, labels, input_dims):

        SH = Shuffler(atom_vecs, labels, input_dims)

        self.ordtrain, self.ordtrain_labels, self.ordtest, self.ordtest_labels = SH.train_test_split()
        self.train, self.train_labels, self.test, self.test_labels = SH.shuff_train_test_split()




    def ff_fitness(self, input_dim, curout_dim, optimizer='sgd', hyperoptimizer='random'):

        with tf.device('/cpu:0'):
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
                current_dim=curout_dim,
                optimize=True,
                seed=24,
                optimizer=optimizer,

                epochs=1,
                initial_points=3,
                hyper_maxepochs=1,
                hyper_iters=1
            )
            xtest = AFF.test_data
            ytest = AFF.test_labels

            if hyperoptimizer == 'random':
                tuner = AFF.random_search()
                best_model = tuner.get_best_models(num_models=1)[0]
                loss = best_model.evaluate(xtest, ytest)  # list[mse loss, mse]
                print('Completed random search\n')
            elif hyperoptimizer == 'bayes':
                tuner = AFF.bayesian()
                best_model = tuner.get_best_models(num_models=1)[0]
                loss = best_model.evaluate(xtest, ytest)  # list[mse loss, mse]
                print('Completed bayesian\n')
            elif hyperoptimizer == 'hyper':
                tuner = AFF.hyperband()
                best_model = tuner.get_best_models(num_models=1)[0]
                loss = best_model.evaluate(xtest, ytest)  # list[mse loss, mse]
                print('Completed hyperband\n')

            # tuner.search_space_summary()


            # loss = [loss_rand[0], loss_bayes[0], loss_hyper[0]]
            # # loss= [loss_bayes[0], loss_hyper[0]]
            # idxloss = loss.index(min(loss))
            # minloss = min(loss)
            # if idxloss == 0:
        return loss[0], best_model, tuner



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


    def PS_integer(self):
        obj = np.zeros((1, self.nvar), dtype='int32')
        for ii in range(self.nvar):
            obj[0, ii] = np.random.randint(self.lb[ii], self.ub[ii])


        objold = np.zeros((1, self.nvar), dtype='int32')
        objnew = np.zeros((1, self.nvar), dtype='int32')
        mu = self.mu0

        oldfit = np.inf
        newfit = 0

        self.build_objdataset(obj)

        curfit, best_model, best_tuner = self.ff_fitness(
            self.input_dims[obj[0, 1]],
            self.curout_dim,
            optimizer=self.optimizers[obj[0, 5]],
            hyperoptimizer=self.hyperoptimizers[obj[0, 7]]
        )



        diff = np.inf
        iter = 0

        while iter < self.maxiter:
            #  and diff > self.mindiff:
            if iter % 10:
                print(f'Begin iteration {iter}')
            changeflag = 0
            mesh = self.make_mesh(obj)
            mesh = self.enforce_bounds(mesh)
            mesh = self.intcons(mesh)

            # check if any of the exploratory points have lower fitness
            for ii in range(1, self.nvar):
                mesh[ii, :] = self.build_objdataset(mesh[ii, :])
                print(mesh[ii, :])
                meshfit, meshmod, tuner = self.ff_fitness(
                    self.input_dims[mesh[ii, 1]],
                    self.curout_dim,
                    optimizer=self.optimizers[mesh[ii, 5]],
                    hyperoptimizer=self.hyperoptimizers[mesh[ii, 7]]
                )

                if meshfit < curfit:
                    curfit = meshfit
                    objnew = mesh[ii, :]
                    objnew = np.reshape(objnew, (1,len(objnew)))
                    best_model = meshmod
                    best_tuner = tuner
                    changeflag = 1


            if changeflag == 1:
                mu = self.mu0
                while newfit < curfit:
                    objold = obj
                    oldfit = curfit
                    obj = objnew
                    objnew = objold + self.alpha*(obj - objold)
                    objnew = self.enforce_bounds(objnew)
                    objnew = self.intcons(objnew)
                    self.build_objdataset(objnew)

                    newfit, _, _ = self.ff_fitness(
                        self.input_dims[objnew[0, 1]],
                        self.curout_dim,
                        optimizer=self.optimizers[objnew[0, 5]],
                        hyperoptimizer=self.hyperoptimizers[objnew[0, 7]]
                    )
                    mesh = self.make_mesh(objnew)
                    mesh = self.enforce_bounds(mesh)
                    mesh = self.intcons(mesh)

                    for ii in range(1, self.nvar):
                        self.build_objdataset(mesh[ii, :])

                        meshfit, meshmod, tuner = self.ff_fitness(
                            self.input_dims[mesh[ii, 1]],
                            self.curout_dim,
                            optimizer=self.optimizers[mesh[ii, 5]],
                            hyperoptimizer=self.hyperoptimizers[mesh[ii, 7]]
                        )
                        if meshfit < curfit:
                            oldfit = curfit
                            curfit = meshfit
                            objnew = mesh[ii, :]
                            best_model = meshmod
                            best_tuner = tuner

            else:
                mu = mu - self.delta
                iter += 1

            print(f'Minimum fitness is {curfit}')

        print(f'Optimal hyperparameters')
        for ii in range(len(obj)):
            print(f'{ii} == {obj[0, ii]}')


        return best_model, best_tuner












if __name__ == '__main__':

    paras_path = Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\raw_ricoparas.pkl')
    with open(paras_path, 'rb') as f:
        paras = pickle.load(f)

    doc_dict_path = Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\doc_dict.pkl')
    with open(doc_dict_path, 'rb') as f:
        doc_dict = pickle.load(f)

    encoded_data_paths = [
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Rico-Corpus\encoded_data_01.pkl'),
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Scientific-Articles\ricocorpus_sciart_encoded.pkl')
    ]
    encoded_data = []
    for en in encoded_data_paths:
        with open(en, 'rb') as f:
            encoded_data.append(pickle.load(f))

    vocab_paths = [
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Rico-Corpus\ranked_vocab.pkl'),
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Scientific-Articles\sciart_vocab.pkl')
    ]
    vocab = []
    for v in vocab_paths:
        with open(v, 'rb') as f:
            vocab.append(pickle.load(f))

    input_dims = [2, 10, 30, 50, 2]
    output_dims = [2, 3, 10] # greatly increases computation time

    weighting_factor = [1, 5, 25, 125, 625]

    optimizers = ['sgd', 'adam', 'adagrad', 'rmsprop', 'adadelta', 'adamax']

    label_paths = [
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Full_Ordered_Labels_2Dims.pkl'),
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Full_Ordered_Labels_3dims.pkl'),
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Full_Ordered_Labels_10Dims.pkl')
                   ]
    labels = []
    for la in label_paths:
        with open(la, 'rb') as f:
            labels.append(pickle.load(f))




    we_model_paths = [
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Rico-Corpus\models\ricocorpus_model10000ep_2dims'),
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Rico-Corpus\models\ricocorpus_model10000ep_10dims'),
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Rico-Corpus\models\ricocorpus_model10000ep_30dims'),
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Rico-Corpus\models\ricocorpus_model10000ep_50dims'),
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Scientific-Articles\sciart_model')
    ]




    atom_embed_method = ['sum_atoms', 'avg_atoms', 'ndelta']


    # for output in output_dims:
    #     for curdim in range(output):
    #         Learn_Master(
    #
    #         )

    hyperoptimizers = ['random', 'bayes', 'hyper']

    ints = [0, 1, 2, 3, 4, 5, 6, 7]

    nn_arch = ['ff']
    curdim = 0
    LM = Learn_Master(
        doc_dict,
        vocab,
        encoded_data,
        labels,
        ints,
        input_dims,
        we_model_paths,
        output_dims,
        weighting_factor,
        optimizers,
        atom_embed_method,
        nn_arch,
        hyperoptimizers,
        curout_dim=curdim,
        mu0=4,
        alpha=1,
        delta=1,
        maxiter=10,
    )

    LM.PS_integer()



