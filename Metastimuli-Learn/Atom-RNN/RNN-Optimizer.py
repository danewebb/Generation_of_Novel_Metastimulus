import tensorflow as tf
from Atom_RNN import Atom_RNN
import numpy as np
import pickle
from tensorflow import keras

import time

# VERSION: keras-tuner==1.0.0

tr_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\BOWsum\w100\train.pkl'
trlabels_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\BOWsum\w100\train_labels.pkl'

te_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\BOWsum\w100\test.pkl'
telabels_path = r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\BOWsum\w100\test_labels.pkl'

set_batch_size = 10
set_epochs = 100
learn_rate = 0.005
numsteps=5 # metaparameter


# LOG_DIR = f'{int(time.time())}'
with tf.device('/cpu:0'):
    AR = Atom_RNN(
        data_path=tr_path,
        train_label_path=trlabels_path,
        test_path=te_path,
        test_label_path=telabels_path,
        epochs=set_epochs,
        # learning_rate=learn_rate,
        steps=numsteps,
        optimize=True,
        seed=24
    )
    # tuner = AR.random_search()
    tuner = AR.bayesian()
    # tuner = AR.hyperband()

    print('hello world')

