import tensorflow as tf
from Atom_FFNN import Atom_FFNN
import numpy as np
import pickle
from tensorflow import keras

import time

# VERSION: keras-tuner==1.0.0

tr_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_30dims\BOWsum_rico\W_100_output_3Dims\train.pkl'
trlabels_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_30dims\BOWsum_rico\W_100_output_3Dims\train_labels.pkl'

te_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_30dims\BOWsum_rico\W_100_output_3Dims\test.pkl'
telabels_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_30dims\BOWsum_rico\W_100_output_3Dims\test_labels.pkl'

set_batch_size = 10
set_epochs = 5
learn_rate = 0.005

LOG_DIR = f'{int(time.time())}'
name = 'random_search'
with tf.device('/cpu:0'):
    AFF = Atom_FFNN(
        data_path=tr_path,
        train_label_path=trlabels_path,
        test_path=te_path,
        test_label_path=telabels_path,
        nullset_path=te_path,
        batch_size=set_batch_size,
        epochs=set_epochs,
        learning_rate=learn_rate,
        dense_out=1,
        dense_in=30,
        current_dim=0,
        optimize=True,
        seed=24
    )
    xtest = AFF.test_data
    ytest = AFF.test_labels

    tuner = AFF.random_search(LOG_DIR, name)
    # tuner = AFF.bayesian()
    # tuner = AFF.hyperband()

    # tuner.search_space_summary()
    best_model = tuner.get_best_models(num_models=1)[0]
    loss = best_model.evaluate(xtest, ytest) # list[mse loss, mse]

    print('hello world')
