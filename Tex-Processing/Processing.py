import numpy as np
import pickle
import os
import re
import nltk
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

    def __init__(self, data=None, data_dir='', vocab=None, vocab_dir=''):


        if data != None:
            self.data = data
        elif data_dir != '':
            with open(data_dir, 'rb') as f1:
                self.data = pickle.load(f1)
        else:
            raise ValueError('No valid data in input')

        self.encoded_data = []

        self.data_lens = []
        # with open(train_projection, 'rb') as train_labels:
        #     self.train_plabels = pickle.load(train_labels)

        # self.train_nodes, self.train_vectors = self.__pims_labels('pims-filter/adjacency_train.json')
        # self.test_nodes, self.test_vectors = self.__pims_labels('pims-filter/')
        if vocab_dir != '':
            with open(vocab_dir, 'rb') as vocab_data:
                self.vocab_data = pickle.load(vocab_data)
        elif vocab != None:
            self.vocab_data = vocab
        else:
            ValueError('Vocab is not defined')







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
        from nltk.stem import WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()

        return lemmatizer.lemmatize(word)


    def remove_stop(self, word):
        stop_words = set(stopwords.words('english'))
        if word in stop_words:
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
    PCS = Tex_Processing(data_dir=r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\raw_ricoparas.pkl',
                         vocab_dir=r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Rico-Corpus\ranked_vocab.pkl')
    PCS.main(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\Rico-Corpus\encoded_data_01.pkl')
