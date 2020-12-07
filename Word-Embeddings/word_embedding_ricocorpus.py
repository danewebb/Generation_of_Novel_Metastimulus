import tensorflow as tf
import numpy as np
import os
# import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

import pickle

# Lemmatize words, https://en.wikipedia.org/wiki/Lemmatisation


# batch_size = 20
# set_epochs = 100
# set_valsteps = 20
# embedding_dim = 2
# dense1_size = 16
# dense2_size = 1


class word_embedding_ricocorpus():


    def __init__(self, encoded_data, ranked_vocab, embedding_dim, set_epochs=100, batch_size=20,
                 dense1_size = 16, dense2_size = 1, shuffling=True, save_location=''):

        self.data = encoded_data
        self.vocab = ranked_vocab
        self.embedding_dim = embedding_dim

        self.vocab_size = len(self.vocab)

        self.set_epochs = set_epochs
        self.batch_size = batch_size
        self.shuffling = shuffling
        self.dense1_size = dense1_size
        self.dense2_size = dense2_size
        self.labels = []
        self.padded_data = 0

        self.save_location = save_location

        self.train = train

    def list_to_numpy(self):
        # non-csv path
        # list -> ndarray
        data_arr = np.array([np.array(x) for x in self.data])

        # post padding
        self.padded_data = keras.preprocessing.sequence.pad_sequences(
            data_arr, padding='post')

        # dummy labels
        tdata_size = np.shape(data_arr)
        labels = np.zeros([tdata_size[0], 1])
        self.labels = labels.astype('int32')

        # convert train_vec into tf dataset. Could be valuable.

        # td = tf.data.Dataset.from_tensors((train_data_arr, trlabels))
        # td = tf.data.Dataset.from_tensors(train_data_arr)
        # td = td.shuffle(1000, reshuffle_each_iteration = True)
        #
        # # td = td.shuffle(1000, reshuffle_each_iteration = True).padded_batch(10, padded_shapes=([None], ()))
        # td = td.batch(batch_size, drop_remainder=True)



    def train_embedding(self):
        with tf.device('/cpu:0'): # gpu or cpu

            model = keras.Sequential([
                layers.Embedding(self.vocab_size, self.embedding_dim),
                layers.GlobalAveragePooling1D(),
                layers.Dense(self.dense1_size, activation='relu'),
                layers.Dense(self.dense2_size)
            ])

            model.summary()


            model.compile(
                optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy']
            )

            history = model.fit(
                self.padded_data, self.labels,
                epochs=self.set_epochs
                , batch_size=self.batch_size
                , shuffle=self.shuffling
            )

            if self.save_location != '':
                # save model
                keras.models.save_model(model, self.save_location)




        # ### Retrieve Word Embeddings
        # e = model.layers[0]
        # weights = e.get_weights()[0]
        # print(weights.shape)
        # #
        # import io
        # #
        # # # need to decode to get words rather than numbers
        # # # encoder = info.features['text'].encoder
        # #
        # out_v = io.open(r'ricocorpus_wordembedding\vecs.tsv', 'w', encoding='utf-8')
        # out_m = io.open(r'ricocorpus_wordembedding\meta.tsv', 'w', encoding='utf-8')
        # #
        # for num, word in enumerate(vocab):
        #   vec = weights[num]
        #   out_m.write(word + "\n")
        #   out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        #
        #
        #
        #
        # out_v.close()
        # out_m.close()




if __name__ == '__main__':
    # clean, encoded training data
    # with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\ricocorpus_wordembedding\train_vec.pkl', 'rb') as file:
    #     train_data = pickle.load(file)
    #
    # # lengths of each atom
    # with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\ricocorpus_wordembedding\train_len.pkl', 'rb') as file1:
    #     train_lens = pickle.load(file1)

    with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\ricocorpus_wordembedding\test_vec.pkl', 'rb') as file:
        data = pickle.load(file)

    # lengths of each atom
    with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\ricocorpus_wordembedding\test_len.pkl', 'rb') as file1:
        lens = pickle.load(file1)

    # vocab in order of most common -> least common
    with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\ricocorpus_wordembedding\ranked_vocab.pkl',
              'rb') as voc_file:
        vocab = pickle.load(voc_file)

    wre = word_embedding_ricocorpus(data, vocab, 2, shuffling=True,
                                    save_location=r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\ricocorpus_wordembedding\ricocorpus_model')

