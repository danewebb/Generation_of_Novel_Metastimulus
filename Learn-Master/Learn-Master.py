from atom_tag_pairing import Atom_Tag_Pairing
from atom_embedding import Atom_Embedder
from Atom_FFNN import Atom_FFNN
from Atom_RNN import Atom_RNN
from word_embedding_ricocorpus import word_embedding_ricocorpus
from word_embedding import Sciart_Word_Embedding

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
    def __init__(self, dataset_dict, vocab, ordered_encdata, NN_prebuilt_model=None, NNclasse=None, WE_prebuilt_model=None,
                 WE_classes=None, vocab_save_path='', data_save_path='',
                 projection=None, adjacency=None,
                 keyword_weight_factor=1,
                 lb=None, ub=None
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

        self.keyword_weigth_factor = keyword_weight_factor

        if lb is not None:
            self.lb = lb
            self.ub = ub

        if adjacency is not None:
            self.adj = adjacency
        if projection is not None:
            self.proj = projection
        if projection is not None and adjacency is not None:
            self.label_pairing()

        if WE_prebuilt_model is not None:
            self.we_model = keras.models.load_model(WE_prebuilt_model)




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


    def atom_embedding(self, atom_embed_method, ndel=8):
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

        return atom_vecs




    def build_word_embedding(self, data, we_batch_size, embed_dim, save_we_model_path, vocab_path, we_model_path=''):
        Word_Embed = Sciart_Word_Embedding(data, batch_size=we_batch_size, embedding_dim=embed_dim,
                                           save_model_path=save_we_model_path, vocab_path=self.vocab_save_path, model_path=we_model_path)
        Word_Embed.main()


    def ff_fitness(self, train_path, trlabels_path, test_path, telabels_path, input_dim, curout_dim):
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


    def PS_integer(self, mu0):
        nvar = len(self.lb)
        xcur = np.zeros(nvar, dtype='int32')
        xold = np.zeros(nvar, dtype='int32')
        xnew = np.zeros(nvar, dtype='int32')
        mu = mu0

        oldfit = np.inf
        newfit = np.inf

        curfit = self.ff_fitness(
            self.list_train_paths[np.random.randint((0, len(self.list_train_paths)))], # lb[0]
            self.list_train_label_paths[np.random.randint((0, len(self.list_train_label_paths)))], # lb[1]
            self.list_test_paths[np.random.randint((0, len(self.list_test_paths)))], # lb[2]
            self.list_test_label_paths[np.random.randint((0, len(self.list_test_label_paths)))], # lb[3]
            self.input_dims[np.random.randint((0, len(self.input_dims)))],
            self.curout_dims[np.random.randint((0, len(self.curout_dims)))]


        )
        for ii in range(nvar):
            print('hellow world')
























