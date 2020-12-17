import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # info and warning messages aren't printed
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import time
import pickle
# import tensorflow_datasets as tfds

# import Process_sciart as PSA

# tfds.disable_progress_bar()


class Sciart_Word_Embedding:

    def __init__(self, data_chunks=None, data=None, model_path='', labels=0, vocab_path='', batch_size=20, set_epochs=1, embedding_dim=2,
                 dense1_size=16, dense2_size=1, save_model_path=''):
        """

        :param data_chunks: List of data file paths. Break up data so memory doesn't overflow.
        :param model: First time through leave blank but once model is saves in path, change, '', to model path.
        :param labels: Dummy labels = 0, every label is a zero.
        :param vocab: Path for corpus vocab. Leave as, '', for scientific papers
        :param vocab_size: If we are using scientific papers vocab built in Process_sciart, leave vocab_size=0

        :param batch_size:
        :param set_epochs: Leave as 1 if running from class.
        :param embedding_dim: Based on PIMS filter dimension output, (e.g. Projection.pkl)
        :param dense1_size: Size of first Dense, 'relu', layer
        :param dense2_size: Size of second Dense 'default', layer
        """

        if data_chunks:
            self.data_chunks = data_chunks
        elif data:
            self.data = data
        if vocab_path == '' or not os.path.exists(vocab_path):
            # get vocab_size from 'sciart_vocab.pkl'
            raise ValueError('vocab_path does not exist')
        else:
            # use another vocab
            with open(vocab_path, 'rb') as voc_file:
                self.vocab = pickle.load(voc_file)

            self.vocab_size = len(self.vocab)

        self.labels = labels

        self.batch_size = batch_size
        self.set_epochs = set_epochs
        self.embedding_dim = embedding_dim
        self.dense1_size = dense1_size
        self.dense2_size = dense2_size

        self.weights = None
        self.embedded_vecs = None

        if model_path == '' or not os.path.exists(model_path):

            self.model = self.build_model()
        else:
            # load different model
            self.model = keras.models.load_model(model_path)

        if save_model_path == '' or not os.path.exists(save_model_path):
            self.save_model_path == ''
            Warning('save_model_path is undefined. This will result in the model not saving.')
        else:
            self.save_model_path = save_model_path

    def retrieve_word_embeddings(self, model_path, vecs_path, meta_path):
        ### Retrieve Word Embeddings
        import io

        # load model
        model = keras.models.load_model(model_path)
        e = model.layers[0]
        weights = e.get_weights()[0] # word embeddings
        # print(weights.shape)

        out_v = io.open(vecs_path, 'w', encoding='utf-8') # vector file
        out_m = io.open(meta_path, 'w', encoding='utf-8') # meta file, words.
        #

        for num, word in enumerate(self.vocab[1:], start=1): # Int 0 is in index 0 of vocab
            vec = weights[num]
            out_m.write(word + "\n")
            out_v.write('\t'.join([str(x) for x in vec]) + "\n")

        # save
        out_v.close()
        out_m.close()

    def data_to_numpy(self, data):
        # make encoded data into numpy arrays
        train_data_arr = np.array([np.array(x) for x in data])  # turns list into a np array

        # array padding with zeros
        padded_train_data = keras.preprocessing.sequence.pad_sequences(
            train_data_arr, padding='post'
        )

        tdata_size = padded_train_data.shape

        return padded_train_data, tdata_size

    def build_model(self):
        # Build model if called
        model = keras.Sequential([
            layers.Embedding(self.vocab_size, self.embedding_dim),
            layers.GlobalAveragePooling1D(),
            layers.Dense(self.dense1_size, activation='relu'),
            layers.Dense(self.dense2_size)
        ])

        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        return model

    def train(self, trdata, trlabels):
        # Training call
        history = self.model.fit(
            trdata,
            trlabels,
            epochs=self.set_epochs,
            batch_size=self.batch_size,
            shuffle=True # shuffles batches
            ,verbose=0 # supresses progress bar
        )


    def main_train(self):
        """
        Assumes self.data_chunks is a list of file paths with clean text. Main loop calls functions within the class to
        convert the data, grab labels, and then train the network. Finally it saves the model when it completes the
        list of chunks in self.data_chunks.
        :return:
        """
        for chunk in self.data_chunks:
            with open(chunk, 'rb') as chunk_file:
                data = pickle.load(chunk_file)
            trdata, tdata_size = self.data_to_numpy(data)
            if self.labels == 0: # if we want to use dummy labels
                trlabels = np.zeros([tdata_size[0], 1])
                trlabels = trlabels.astype('int32')
            else:
                trlabels = self.labels # want to train both NN and word embedding

            self.train(trdata, trlabels) # Training call

        if self.save_model_path != '':
            tf.keras.models.save_model(self.model, self.save_model_path)
        else:
            print('Model was not saved due to save_model_path being undefined.')


    def encoded_to_embed(self):
        e = self.model.layers[0]
        self.weights = e.get_weights()[0] # word embeddings
        for num, word in enumerate(self.vocab):
            self.embedded_vecs[num] = self.weights[num]


    # def embed_atoms(self):
    #     for word in self.data:

    def word_to_vec(self):
        """
        Converts words into vector form. This is done by replacing the word with its place in a ranked
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

if __name__ == '__main__':
    with tf.device('/cpu:0'):
        epochs = 12
        report_every = 1
        # data_chunks = [r'Lib/sciart_wordembedding/sciart_data_01.pkl', r'Lib/sciart_wordembedding/sciart_data_02.pkl', r'Lib/sciart_wordembedding/sciart_data_03.pkl',
        #                r'Lib/sciart_wordembedding/sciart_data_04.pkl', r'Lib/sciart_wordembedding/sciart_data_05.pkl', r'Lib/sciart_wordembedding/sciart_data_06.pkl',
        #                r'Lib/sciart_wordembedding/sciart_data_07.pkl', r'Lib/sciart_wordembedding/sciart_data_08.pkl', r'Lib/sciart_wordembedding/sciart_data_09.pkl',
        #                r'Lib/sciart_wordembedding/sciart_data_10.pkl', r'Lib/sciart_wordembedding/sciart_data_11.pkl', r'Lib/sciart_wordembedding/sciart_data_12.pkl']

        data_chunks = [r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_data_folder\sciart_data_01.pkl',
                       r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_data_folder\sciart_data_02.pkl',
                       r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_data_folder\sciart_data_03.pkl',
                       r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_data_folder\sciart_data_04.pkl',
                       r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_data_folder\sciart_data_05.pkl',
                       r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_data_folder\sciart_data_06.pkl',
                       r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_data_folder\sciart_data_07.pkl',
                       r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_data_folder\sciart_data_08.pkl',
                       r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_data_folder\sciart_data_09.pkl',
                       r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_data_folder\sciart_data_10.pkl',
                       r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_data_folder\sciart_data_11.pkl',
                       r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_data_folder\sciart_data_12.pkl'
        ]

        SWE = Sciart_Word_Embedding(data_chunks, model_path=r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_wordembedding\sciart_model',
                                    vocab_path=r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_wordembedding\sciart_vocab.pkl',
                                    save_model_path=r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_wordembedding\sciart_model')




        start_time = time.time()
        for ii in range(1, epochs+1):
            SWE.main()
            if ii%report_every == 0:
                end_time = time.time()
                print(f'{ii}/{epochs} completed')

                elapsed_time = end_time - start_time
                reports_left = (epochs - ii)/report_every
                eta = elapsed_time*reports_left/3600 ### Hours
                print(f'Finished epoch {ii} at {time.asctime(time.localtime(time.time()))}')
                print(f'Estimated time to completion: {eta} hours')
                print('\n')
                start_time = end_time


    SWE.retrieve_word_embeddings(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_wordembedding\sciart_model') # extract word embeddings


