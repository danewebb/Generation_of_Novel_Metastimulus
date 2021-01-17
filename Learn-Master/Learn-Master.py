from atom_tag_pairing import Atom_Tag_Pairing
from atom_embedding import Atom_Embedder
from Atom_FFNN import Atom_FFNN
from Atom_RNN import Atom_RNN
# from word_embedding_ricocorpus import word_embedding_ricocorpus
# from word_embedding import Sciart_Word_Embedding

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

class Learn_Master():
    """

    """
    def __init__(self, dataset_dict, vocab, ordered_encdata, ints, curout_dim, NN_prebuilt_model=None, NNclasse=None, WE_prebuilt_model=None,
                 WE_classes=None, vocab_save_path='', data_save_path='',
                 projection=None, adjacency=None,
                 keyword_weight_factor=1,
                 lb=None, ub=None,
                 maxiter=100, mindiff=1e-6, mu0=4,
                 ):
        """
        let us assume dataset is already created and encoded.



        :param dataset:
        :param neural_network:
        :param dataset_type:
        :param NNmodel_path:
        :param WEmodel_path:
        :param vocab_save_path:
        :param data_save_path:
        """


        self.dataset = dataset_dict
        self.vocab = vocab
        self.ordered_encdata = ordered_encdata
        self.labels = None
        self.ints = ints
        self.keyword_weigth_factor = keyword_weight_factor


        self.curout_dim = curout_dim

        if lb is not None:
            self.lb = lb
            self.ub = ub
            self.nvar = len(lb)



        if adjacency is not None:
            self.adj = adjacency
        if projection is not None:
            self.proj = projection
        if projection is not None and adjacency is not None:
            self.label_pairing()


        if WE_prebuilt_model is not None:
            self.we_model = keras.models.load_model(WE_prebuilt_model)


        self.mindiff = mindiff
        self.maxiter = maxiter
        self.mu0= mu0
        self.mu = mu0

        # need to define later
        self.list_train_paths = None
        self.list_train_label_paths = None
        self.list_test_paths = None
        self.list_test_label_paths = None





    def label_pairing(self):
        from atom_tag_pairing import Atom_Tag_Pairing

        if self.adj is not None:
            ATP = Atom_Tag_Pairing(self.dataset, adjacency=self.adj, projection=self.proj)
        else:
            ATP = Atom_Tag_Pairing(self.dataset, projection=self.proj)
        ATP.tag_pairing()
        self.labels = ATP.projection_vectors()


    def atom_embedding(self, atom_embed_method, we_embed, ndel=8):
        atom_vecs = []

        from atom_embedding import Atom_Embedder
        from Weighting_Keywords import Weighting_Keyword

        AE = Atom_Embedder(self.we_model.layers[0].get_weights()[0], self.vocab)
        WK = Weighting_Keyword(self.vocab, self.keyword_weigth_factor)
        WK.keywords_in_vocab()


        if atom_embed_method == 'sum_atoms':
            for para in self.ordered_encdata:
                if para:
                    weights = WK.keyword_search(para)
                    atom_vecs.append(AE.sum_atoms(para, weights=weights))
                else:
                    para = [0]
                    atom_vecs.append(AE.sum_atoms(para))
        elif atom_embed_method == 'avg_atoms':
            for para in self.ordered_encdata:
                if para:
                    weights = WK.keyword_search(para)
                    atom_vecs.append(AE.avg_atoms(para, weights=weights))
                else:
                    para = [0]
                    atom_vecs.append(AE.avg_atoms(para))
        elif atom_embed_method == 'ndelta':
            for para in self.ordered_encdata:
                if para:
                    weights = WK.keyword_search(para)
                    atom_vecs.append(AE.sum_of_ndelta(para, ndel, weights=weights))
                else:
                    para = [0]
                    atom_vecs.append(AE.sum_of_ndelta(para, ndel))

    #     elif atom_embed_method == 'SIF'

        with open(Path(f'C:\\Users\\liqui\\PycharmProjects\\Generation_of_Novel_Metastimulus\\Lib\\Atom-Embeddings\\'
                       f'embeddings\\{we_embed}_w{self.keyword_weigth_factor}_in{atom_vecs[0].shape[0]}_{atom_embed_method}.pkl'),
                  'wb') as f:
            pickle.dump(atom_vecs, f)

        return atom_vecs


    def shuffling(self, atom_vecs, labels):
        from Randomizer import Shuffler
        SH = Shuffler(atom_vecs, labels)

    # def build_word_embedding(self, data, we_batch_size, embed_dim, save_we_model_path, vocab_path, we_model_path=''):
    #     Word_Embed = Sciart_Word_Embedding(data, batch_size=we_batch_size, embedding_dim=embed_dim,
    #                                        save_model_path=save_we_model_path, vocab_path=self.vocab_save_path, model_path=we_model_path)
    #     Word_Embed.main()


    def ff_fitness(self, train_path, trlabels_path, test_path, telabels_path, input_dim, curout_dim,
                   atom_method, word_embed, hyper_optimizer):
        # LOG_DIR = f'{int(time.time())}'
        # name = 'random_search'
        with tf.device('/cpu:0'):
            AFF = Atom_FFNN(
                data_path=train_path,
                train_label_path=trlabels_path,
                test_path=test_path,
                test_label_path=telabels_path,
                nullset_path=test_path,
                # batch_size=set_batch_size,
                # epochs=set_epochs,
                # learning_rate=learn_rate,
                dense_out=1,
                dense_in=input_dim,
                current_dim=curout_dim,
                optimize=True,
                seed=24
            )
            xtest = AFF.test_data
            ytest = AFF.test_labels

            tuner_rand = AFF.random_search()
            tuner_bayes = AFF.bayesian()
            tuner_hyper = AFF.hyperband()

            # tuner.search_space_summary()
            best_model_rand = tuner_rand.get_best_models(num_models=1)[0]
            loss_rand = best_model_rand.evaluate(xtest, ytest)  # list[mse loss, mse]

            best_model_bayes = tuner_bayes.get_best_models(num_models=1)[0]
            loss_bayes = best_model_bayes.evaluate(xtest, ytest)  # list[mse loss, mse]

            best_model_hyper = tuner_hyper.get_best_models(num_models=1)[0]
            loss_hyper = best_model_hyper.evaluate(xtest, ytest)  # list[mse loss, mse]

            loss = [loss_rand[0], loss_bayes[0], loss_hyper[0]]
            minloss = min(loss)
        return minloss



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
        for ii in range(x.shape[1]):
            for int_idx in self.ints:
                xnew[ii, int_idx] = round(x[ii, int_idx])
                if xnew[ii, int_idx] < self.lb[int_idx]:
                    xnew[ii, int_idx] = self.lb[int_idx]
                elif xnew[ii, int_idx] > self.ub[int_idx]:
                    xnew[ii, int_idx] = self.ub[int_idx]

        return xnew


    def enforce_bounds(self, x):
        xnew = np.zeros(x.shape)
        for ii in range(x.shape[1]):
            for jj in range(x.shape[0]):
                if x[ii, jj] < self.lb[jj]:
                    xnew[ii, jj] = self.lb[jj]
                elif x[ii, jj] > self.ub[jj]:
                    xnew[ii, jj] = self.ub[jj]
                else:
                    xnew[ii, jj] = x[ii, jj]

        return xnew


    def PS_integer(self, mu0):
        xcur = np.zeros(self.nvar, dtype='int32')
        xold = np.zeros(self.nvar, dtype='int32')
        xnew = np.zeros(self.nvar, dtype='int32')
        mu = mu0

        oldfit = np.inf
        newfit = np.inf

        curfit = self.ff_fitness(
            self.list_train_paths[np.random.randint((0, len(self.list_train_paths)))], # lb[0]
            self.list_train_label_paths[np.random.randint((0, len(self.list_train_label_paths)))], # lb[1]
            self.list_test_paths[np.random.randint((0, len(self.list_test_paths)))], # lb[2]
            self.list_test_label_paths[np.random.randint((0, len(self.list_test_label_paths)))], # lb[3]
            self.input_dims[np.random.randint((0, len(self.input_dims)))],
            self.curout_dims[np.random.randint((0, len(self.curout_dims)))],
            self.atom_method[np.random.randint((0, len(self.atom_method)))],
            self.word_embed[np.random.randint((0, len(self.word_embed)))],
            self.hyper_optimizer[np.random.randint((0, len(self.hyper_optimizer)))]


        )



        diff = np.inf
        iter = 0
        while iter < self.maxiter and diff > self.mindiff:
            mesh = self.make_mesh(xcur)
            mesh = self.enforce_bounds(mesh)
            mesh = self.intcons(mesh)

            # check if any of the exploratory points have lower fitness
            for ii in range(1, self.nvar):
                meshfit = self.ff_fitness(
                    self.list_train_paths[mesh[ii, 0]],  # lb[0]
                    self.list_train_label_paths[mesh[ii, 1]],  # lb[1]
                    self.list_test_paths[mesh[ii, 2]],  # lb[2]
                    self.list_test_label_paths[mesh[ii, 3]],  # lb[3]
                    self.input_dims[mesh[ii, 4]],
                    self.curout_dims[mesh[ii, 5]],
                    self.atom_method[mesh[ii, 6]],
                    self.word_embed[mesh[ii, 7]],
                    self.hyper_optimizer[mesh[ii, 8]]
                )












if __name__ == '__main__':

    paras = Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\raw_ricoparas.pkl')
    doc_dict = Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\doc_dict.pkl')

    encoded_data = [
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Rico-Corpus\encoded_data_01.pkl'),
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Scientific-Articles\ricocorpus_sciart_encoded.pkl')
    ]

    vocab = [
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Rico-Corpus\ranked_vocab.pkl'),
        Path(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Scientific-Articles\sciart_vocab.pkl')
    ]

    input_dims = [2, 10, 30, 50]
    output_dims = [2, 3, 10] # greatly increases computation time

    weighting_factor = [1, 5, 25, 125, 625]















