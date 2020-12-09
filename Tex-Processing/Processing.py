import numpy as np
import pickle
import os
import re
import json
import tensorflow as tf
# import tensorflow_datasets as tfds
# from tensorflow import keras
# from tensorflow.keras import layers


class Tex_Processing():
    """
    This class handles splitting data between
    """

    def __init__(self, training_data_dir, testing_data_dir, vocab_dir):
        with open(training_data_dir, 'rb') as training_data:
            self.train_data =  pickle.load(training_data)

        with open(testing_data_dir, 'rb') as testing_data:
            self.test_data = pickle.load(testing_data)


        # with open(train_projection, 'rb') as train_labels:
        #     self.train_plabels = pickle.load(train_labels)

        # self.train_nodes, self.train_vectors = self.__pims_labels('pims-filter/adjacency_train.json')
        # self.test_nodes, self.test_vectors = self.__pims_labels('pims-filter/')

        with open(vocab_dir, 'rb') as vocab_data:
            self.vocab_data = pickle.load(vocab_data)



        self.xtrain = []
        self.ytrain = []

        self.xtest = []
        self.ytest = []

        self.embedxtrain = []
        self.embedxtest = []

        self.train_lens = []
        self.test_lens = []



    def split_data(self):
        """
        grab paragraphs from the master dictionary.
        :return:
        """

        if self.train_data == None or self.test_data == None:
            ValueError('Both training and testing datasets must exist.')

        if self.train_data is not dict:
            TypeError('training_data argument must be a dictionary.')


        for value in self.train_data.values():
            paradict = value['para_dict']
            self.xtrain.append(paradict['paragraph'])

        for value in self.test_data.values():
            paradict = value['para_dict']
            self.xtest.append(paradict['paragraph'])




    def random_idx(self, random_state=24):
        # randomize the data and labels. Keeps them paired but changes the indices of each paragraph/label combo
        np.random.seed(random_state)

        train_len = len(self.xtrain)
        train_idx = list(range(train_len))
        np.random.shuffle(train_idx)

        # pre-allocate lists
        new_xtrain = [None]*train_len
        new_ytrain = [None]*train_len

        for ii, idx in enumerate(train_idx):
            new_xtrain[ii] = self.xtrain[idx]
            new_ytrain[ii] = self.train_vectors[idx]

        test_len = len(self.xtest)
        test_idx = list(range(test_len))
        np.random.shuffle(test_idx)
        new_xtest = [None]*test_len
        new_ytest = [None]*test_len

        for ii, idx in enumerate(test_idx):
            new_xtest[ii] = self.xtest[idx]
            new_ytest[ii] = self.test_vectors[idx]


        # print(f'self.random_idx ending lengths\n'
        #       f'xtrain length = {len(new_xtrain)}, ytrain length = {len(new_xtrainhot)}\n'
        #       f'xtest length = {len(new_xtest)}, ytest length = {len(new_xtesthot)}\n')

        return new_xtrain, new_ytrain, new_xtest, new_ytest


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


        for paras in self.xtrain: # loops each item in the list

            for para in paras: # takes the paragraph from the item
                for let in para: # each letter in paragraph
                    # if letter is a space, everything previous makes up a word
                    if let == ' ':
                        word = ''.join(word_store)
                        clean_word = self.word_cleaner(word)
                        word_store = []

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

            self.embedxtrain.append(para_vec)
            para_vec = []




        word_store = []
        for paras in self.xtest:
            for para in paras:
                for let in para:
                    if let == ' ':
                        word = ''.join(word_store)
                        clean_word = self.word_cleaner(word)
                        word_store = []

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

            self.embedxtest.append(para_vec)
            para_vec = []



    def atom_embeddings(self):
        """
        Embed atoms by adding up the vectors of the constituent words
        :return:
        """
        embedatom_train = []
        embedatom_test = []
        for para in self.embedxtrain:
            vec = 0
            for word in para:
                vec += word
            embedatom_train.append(vec)


        for para in self.embedxtest:
            vec = 0
            for word in para:
                vec += word
            embedatom_test.append(vec)

        return embedatom_train, embedatom_test



    def vec_lengths(self):
        ele_len = 0
        for item in self.embedxtrain:
            for ele in item:
                ele_len += 1
            self.train_lens.append(ele_len)
            ele_len = 0

        for item in self.embedxtest:
            for ele in item:
                ele_len += 1
            self.test_lens.append(ele_len)
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





    def main(self, train_vecs_path, test_vecs_path, threshold=1):

        # self.pims_labels()

        output_dict = dict()
        vocab = self.handle_vocab(threshold)

        self.split_data()
        # Lets not make these class variables to

        # Randomize indices while keeping labels and para together.
        # train_para, train_lab, test_para, test_lab = self.random_idx(ytrain, ytest, random_state=random_state)


        self.word_to_vec(vocab)

        # with open('ranked_vocab.pkl', 'wb') as rankvoc_file:
        #     pickle.dump(vocab, rankvoc_file)
        #
        # with open('train_vec.pkl', 'wb') as file:
        #     pickle.dump(self.embedxtrain, file)

        with open(train_vecs_path, 'wb') as train_vec_file:
            pickle.dump(self.embedxtrain, train_vec_file)

        with open(test_vecs_path, 'wb') as test_vec_file:
            pickle.dump(self.embedxtest, test_vec_file)

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
    PCS = Data_Processing('training_dict.pkl', 'testing_dict.pkl', 'rank_vocab.pkl', 'pims-filter/Projection.pkl')
    PCS.main()
