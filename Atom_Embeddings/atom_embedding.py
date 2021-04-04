import numpy as np
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import pickle
# import tensorflow as tf
# from sklearn.decomposition import TruncatedSVD
import keras

import gensim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class Atom_Embedder:

    def __init__(self, weights, vocab):
        # require each paragraph to be fed in to class?

        # self.atoms = listof_atoms # discrete unit of text, e.g. chapters, paragraphs, sentences.
        self.weights = weights # model weights from word embeddings
        self.vocab = vocab


        # dictionary of word encoded values as keys, and the vectors as values
        self.encoded_vecs = dict()
        self.__assign_vecs_to_enc()


        self.pvdm_model = None


    def __assign_vecs_to_enc(self):
        # Assigns corresponding vectors to the encoded word values.
        for num, word in enumerate(self.vocab):  # 0 index of vocab is 0????
            # loop through vocab and assign the correct weights to the vocab
            self.encoded_vecs[num] = self.weights[num]


    def __init_diffs(self, atom):
        diffs = []
        for ii in range(0, len(atom) - 1):
            # self.encoded_vecs: keys=encoded word, value=word embed vector
            diffs.append(self.encoded_vecs[atom[ii + 1]] - self.encoded_vecs[atom[ii]])

        return diffs

    def sum_of_difference(self, atom):
        """
        :param atom: One atom per call
        :return: result
        (1, 2, 3), (4, 5, 6), (7, 8, 9)
        ((4-1), (5-2), (6-3)) + ((7-4), (8-5), (9-6))
        """

        diffs = self.__init_diffs(atom)

        # sums all resultant vectors and collapses list
        res = [sum(jj) for jj in zip(*diffs)]

        return res

    def sum_of_ndelta(self, atom, n, weights=[]):
        """
        n = 1: difference between each vector than sum
        n = 2: difference between each vector then a difference between resultant then sum.
        n = 3, 4, 5, ...
        :param atom:
        :param n: Number of deltas
        :return:
        """
        if weights == []: # not quite sure how to do these weights with the ndelta
            weights = np.ones(len(atom))
        diff1 = []
        diff2 = self.__init_diffs(atom)
        L = len(diff2)
        n -= 1
        while n > 0 and L > 1:
            for ii in range(0, L - 1):
                diff1.append(diff2[ii + 1] - diff2[ii])
            n -= 1
            L = len(diff1)
            diff2 = diff1
            diff1 = []

        res = [sum(jj) for jj in zip(*diff2)]
        return res


    def sum_atoms(self, atom, weights=[]):
        """
        sums one atom per call
        :param atom: One atom per call
        :return: result
        (1, 2, 3) + (4, 5, 6) = (5, 7, 9)
        """
        sums = []
        if weights == []:
            weights = np.ones(len(atom))
        for ii in range(0, len(atom)): # 0  or -1 ???
            sums.append(self.encoded_vecs[atom[ii]] * weights[ii])

        res = [sum(jj) for jj in zip(*sums)]

        return res

    def avg_atoms(self, atom, weights=[]):
        """
        averages one atom per call
        :param atom:
        :return: result
        (1, 2, 3), (4, 5, 6) = (2.5, 3.5, 4.5)
        """
        atom_vectors = []
        if weights == []:
            weights = np.ones(len(atom))
        for ii in range(0, len(atom)):
            atom_vectors.append(self.encoded_vecs[atom[ii]]*weights[ii])

        atom_arr = np.asarray(atom_vectors)

        avgs = np.average(atom_arr, axis=0)

        return avgs

    def __read_corpus(self, raw_paras, tag_bool=False):
        tokens = None
        tagged = []
        if tag_bool:
            for ii, lines in enumerate(raw_paras):
                line = ' '.join(lines)
                tokens = gensim.utils.simple_preprocess(line)
                tagged.append(gensim.models.doc2vec.TaggedDocument(tokens, [ii]))
        else:
            tagged = None
            line = ' '.join(raw_paras)
            tokens = gensim.utils.simple_preprocess(line)

        return tokens, tagged



    def pvdm_train(self, raw_paras):
        vec_size = self.weights.shape[1]
        _, corpus = list(self.__read_corpus(raw_paras, tag_bool=True))
        self.pvdm_model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_size, min_count=2, epochs=1000, dm=1)
        self.pvdm_model.build_vocab(corpus)

        self.pvdm_model.train(corpus, total_examples=self.pvdm_model.corpus_count, epochs=self.pvdm_model.epochs)


    def pvdm(self, raw_para, weight=1):
        para, _ = self.__read_corpus(raw_para)
        vector = self.pvdm_model.infer_vector(para)

        vector = vector*weight
        return vector


    # todo: SIF atom embedding
    # SIF atom embedding section
    def weighted_avg(self, atom, embed_dim):
        numw = len(atom) # number of words in an atom
        senw = np.zeros((numw, embed_dim))
        emb = np.zeros((numw, embed_dim))
        # atom == x
        for jj in range(numw):
            senw[jj, :] = self.encoded_vecs[atom[jj]] # w
        for ii in range(numw):
            word_vec = self.encoded_vecs[atom[ii]] # We
            emb[ii, :] = senw[ii, :].dot(word_vec)/np.count_nonzero(senw[ii, :]) # just a copy of the same float

        return emb

    def comp_pc(self, c0, npc, num):
        return np.random.randint
        # svd = TruncatedSVD(n_components=npc, n_iter=num, random_state=0)
        # svd.fit(c0)
        # return svd.components_

    def remove_pc(self, c0, npc, num_iter):
        pc = self.comp_pc(c0, npc, num_iter)
        if npc == 1:
            cc0 = c0 - c0.dot(pc.transpose())*pc
        else:
            cc0 = c0 - c0.dot(pc.transpose()).dot(pc)

        return cc0

    def SIF_embedding(self, atom, embed_dim, npc, num_iter, weights=None):
        emb = self.weighted_avg(atom, embed_dim)

        emb = self.remove_pc(emb, npc, num_iter)

        return emb








if __name__ == '__main__':


    from Lib.Word_Embeddings.Weighting_Keywords import Weighting_Keyword
    atom_vecs = []
    weighting_factor = 100
    ndel = 8
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\raw_ricoparas.pkl', 'rb') as f0:
        raw_paras = pickle.load(f0)

    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word_Embeddings\Rico-Corpus\ranked_vocab.pkl', 'rb') as voc_file:
        vocab = pickle.load(voc_file)

    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word_Embeddings\Rico-Corpus\encoded_data_01.pkl', 'rb') as vec_file:
        encoded = pickle.load(vec_file)

    model = keras.models.load_model(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word_Embeddings\Rico-Corpus\models\ricocorpus_model10000ep_30dims')

    AE = Atom_Embedder(model.layers[0].get_weights()[0], vocab)
    AE.pvdm_train(raw_paras)

    WK = Weighting_Keyword(vocab, weighting_factor)
    WK.keywords_in_vocab()





    for para in encoded:
        # atom_vecs.append(AE.sum_of_difference(para))


        if para:
            weights = WK.keyword_search(para)
            atom_vecs.append(AE.sum_atoms(para, weights=weights))
        else:
            # There is one empty paragraph. Error, grabbed latex code and then cleaned it.
            para = [0]
            atom_vecs.append(AE.sum_atoms(para))

    # saving

    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\BOWsum\all_atoms_weighted_100alpha.pkl', 'wb') as file:
        pickle.dump(atom_vecs, file)




