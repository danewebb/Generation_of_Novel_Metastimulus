import numpy as np
import pickle
import os
import re
from nltk import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import json
import tensorflow as tf
# import tensorflow_datasets as tfds
# from tensorflow import keras
# from tensorflow.keras import layers


class Tex_Processing():
    """
    This class handles splitting data between
    """

    def __init__(self, data=None, data_dir='', vocab=None, vocab_dir='', doc_dict=None):


        if data is not None:
            self.data = data
        elif data_dir != '':
            with open(data_dir, 'rb') as f1:
                self.data = pickle.load(f1)

        self.vocab_tab = dict()
        self.encoded_data = []
        self.data_lens = []

        self.pattern = r'[a-z]+'

        self.stop_words = set(stopwords.words('english'))

        self.lemmatizer = WordNetLemmatizer()

        # with open(train_projection, 'rb') as train_labels:
        #     self.train_plabels = pickle.load(train_labels)

        # self.train_nodes, self.train_vectors = self.__pims_labels('pims-filter/adjacency_train.json')
        # self.test_nodes, self.test_vectors = self.__pims_labels('pims-filter/')
        if vocab_dir != '':
            with open(vocab_dir, 'rb') as vocab_data:
                self.vocab_data = pickle.load(vocab_data)
        elif vocab is not None:
            self.vocab_data = vocab
        else:
            ValueError('Vocab is not defined')


        if doc_dict is not None:
            self.paras, self.tags = self.dict_breakout(doc_dict)


    def dict_breakout(self, doc_dict):
        paras = []; tags = []
        for ii in range(1, 579+1):
            dic = doc_dict[ii]
            para = dic['para_dict']
            tag = dic['tag_dict']
            paras.append(para['paragraph'])
            tags.append(tag['tags'])
        return paras, tags


    def tokenize(self, string):
        # returns a tokenized an lower cased string
        string = string.lower()
        tok = word_tokenize(string, 'English')
        return tok

    def remove_numsym(self, word):
        w = re.match(self.pattern, word)
        if w:
            return w.string


    def clean_word(self, word):
        w = self.remove_numsym(word)
        if w is not None:
            lem = self.lemmatize(w)
            return lem
        else:
            return None

    def build_vocab(self):
        for para in self.paras:
            for sen in para:
                tokens = self.tokenize(sen)
                for word in tokens:
                    lem = self.clean_word(word)
                    if lem is not None:
                        if lem in self.vocab_tab.keys():
                            self.vocab_tab[lem] += 1
                        else:
                            self.vocab_tab[lem] = 1

        return self.vocab_tab

    def sort_tuple(self, tup, sortbyidx):
        tup.sort(key=lambda x:x[sortbyidx])
        tup.reverse()
        return tup

    def reduce_tuple(self, tup, idxwant):
        want = []
        for ele in tup:
            want.append(ele[idxwant])
        return want

    def rank_vocab(self, vocab_dict, cutoff):
        vocab_tup = []
        culled_vocab = []
        placeholder = list('0')
        for key, val in vocab_dict.items():
            vocab_tup.append((key, val))

        ranked_vocab = self.sort_tuple(vocab_tup, 1)
        for ele in ranked_vocab:
            if ele[1] >= cutoff:
                culled_vocab.append(ele)
        rank_vocab = self.reduce_tuple(culled_vocab, 0)
        full_vocab = placeholder +  rank_vocab
        return full_vocab



    def encode_paras(self, ranked_vocab):
        atom = []
        atoms = []
        for para in self.paras:
            for sen in para:
                tokens = self.tokenize(sen)
                for word in tokens:
                    lem = self.clean_word(word)
                    if lem is not None:
                        nostop = self.remove_stop(lem)
                        if nostop == '':
                            atom.append(0)
                        else:
                            try:
                                atom.append(ranked_vocab.index(nostop))
                            except ValueError:
                                atom.append(0)
            atoms.append(atom)
            atom = []

        return atoms

    # def split_data(self):
    #     """
    #     grab paragraphs from the master dictionary.
    #     :return:
    #     """
    #
    #     if self.train_data == None or self.test_data == None:
    #         ValueError('Both training and testing datasets must exist.')
    #
    #     if self.train_data is not dict:
    #         TypeError('training_data argument must be a dictionary.')
    #
    #
    #     for value in self.train_data.values():
    #         paradict = value['para_dict']
    #         self.xtrain.append(paradict['paragraph'])
    #
    #     for value in self.test_data.values():
    #         paradict = value['para_dict']
    #         self.xtest.append(paradict['paragraph'])



    #
    # def random_idx(self, random_state=24):
    #     # randomize the data and labels. Keeps them paired but changes the indices of each paragraph/label combo
    #     np.random.seed(random_state)
    #
    #     train_len = len(self.xtrain)
    #     train_idx = list(range(train_len))
    #     np.random.shuffle(train_idx)
    #
    #     # pre-allocate lists
    #     new_xtrain = [None]*train_len
    #     new_ytrain = [None]*train_len
    #
    #     for ii, idx in enumerate(train_idx):
    #         new_xtrain[ii] = self.xtrain[idx]
    #         new_ytrain[ii] = self.train_vectors[idx]
    #
    #     test_len = len(self.xtest)
    #     test_idx = list(range(test_len))
    #     np.random.shuffle(test_idx)
    #     new_xtest = [None]*test_len
    #     new_ytest = [None]*test_len
    #
    #     for ii, idx in enumerate(test_idx):
    #         new_xtest[ii] = self.xtest[idx]
    #         new_ytest[ii] = self.test_vectors[idx]


        # print(f'self.random_idx ending lengths\n'
        #       f'xtrain length = {len(new_xtrain)}, ytrain length = {len(new_xtrainhot)}\n'
        #       f'xtest length = {len(new_xtest)}, ytest length = {len(new_xtesthot)}\n')

        # return new_xtrain, new_ytrain, new_xtest, new_ytest


    def handle_vocab(self, threshold):
        # remove uncommon words
        vocab = []
        for word, value in self.vocab_data:
            if value > threshold:
                vocab.append(word)
        vocab.reverse()
        return vocab

    def word_to_vec(self, vocab):
        """
        Converts words in xtrain and xtest into vector form. This is done by replacing the word with its place in a ranked
        vocab list. This is the first step into word embeddings
        :return:
        """

        word_store = []

        para_vec = []


        for paras in self.data: # loops each item in the list

            for para in paras: # takes the paragraph from the item
                for let in para: # each letter in paragraph
                    # if letter is a space, everything previous makes up a word
                    if let == ' ':
                        word = ''.join(word_store)
                        cleaner_word = self.word_cleaner(word)
                        word_store = []

                        nostop = self.remove_stop(cleaner_word)
                        clean_word = self.lemmatize(nostop)

                        for idx, w in enumerate(vocab):
                            if w == clean_word:
                                para_vec.append(idx)
                                break
                            elif w == word:
                                para_vec.append(idx)
                                break
                            elif idx == len(vocab) - 1:
                                para_vec.append(0)
                                break


                    else:
                        word_store.append(let)

            self.encoded_data.append(para_vec)
            para_vec = []


    def vec_lengths(self):
        ele_len = 0
        for item in self.data:
            for ele in item:
                ele_len += 1
            self.data_lens.append(ele_len)
            ele_len = 0


    def word_cleaner(self, word):
        pattern = re.compile(r'\w')
        word_store = []
        for let in word:
            if pattern.match(let):
                word_store.append(let)


        return ''.join(word_store)

    # def __pims_labels(self, file):
    #     with open(file, 'rb') as adjacency:
    #         adj = json.load(adjacency)
    #         nodes = adj['nodes']
    #         vectors = adj['list']
    #         return nodes, vectors


    def save_files(self, data_save_path=''):
        if data_save_path!='':
            with open(data_save_path, 'wb') as f1:
                pickle.dump(self.data, f1)


    def lemmatize(self, word):
        return self.lemmatizer.lemmatize(word)


    def remove_stop(self, word):
        if word in self.stop_words:
            return ''
        else:
            return word


    def main(self, encoded_data_path, threshold=1):

        # self.pims_labels()

        output_dict = dict()
        # vocab = self.handle_vocab(threshold)
        # Lets not make these class variables to

        # Randomize indices while keeping labels and para together.
        # train_para, train_lab, test_para, test_lab = self.random_idx(ytrain, ytest, random_state=random_state)


        self.word_to_vec(self.vocab_data)

        # with open('ranked_vocab.pkl', 'wb') as rankvoc_file:
        #     pickle.dump(vocab, rankvoc_file)
        #
        # with open('train_vec.pkl', 'wb') as file:
        #     pickle.dump(self.embedxtrain, file)

        with open(encoded_data_path, 'wb') as file:
            pickle.dump(self.encoded_data, file)

        # embedatom_train, embedatom_test = self.atom_embeddings()
        # self.vec_lengths()
        #
        # with open('test_len.pkl', 'wb') as test_len_file:
        #     pickle.dump(self.test_lens, test_len_file)

        # with open('train_len.pkl', 'wb') as file1:
        #     pickle.dump(self.train_lens, file1)
        # xtrain, ytrain, xtest, ytest = self.list_to_ndarray(ytrain, ytest)


        # output_dict = {'trainx': self.embedxtrain,  'testx': self.embedxtest,
        #                'voc': vocab, 'trainlen': self.train_lens, 'testlen': self.test_lens, 'atomtrain': embedatom_train,
        #                'atomtest': embedatom_test}

        # 'trainy': train_plab, 'testy': test_plab,

        # return output_dict







if __name__ == '__main__':
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\doc_dict.pkl', 'rb') as f:
        doc_dict = pickle.load(f)
    PCS = Tex_Processing(data_dir=r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\raw_ricoparas.pkl',
                         vocab_dir=r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word_Embeddings\Rico-Corpus\ranked_vocab.pkl',
                         doc_dict=doc_dict
                         )

    vocab_dict = PCS.build_vocab()
    ranked_vocab = PCS.rank_vocab(vocab_dict, 2)
    enc_atoms = PCS.encode_paras(ranked_vocab)

    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Tex-Processing\ranked_vocab_v02.pkl', 'wb') as f:
        pickle.dump(ranked_vocab, f)

    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Tex-Processing\rico_encodings_v02.pkl', 'wb') as f:
        pickle.dump(enc_atoms, f)

    # PCS.main(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Rico-Corpus\encoded_data_02.pkl')
