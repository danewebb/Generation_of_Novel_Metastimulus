
from Graphing import Graphing


import numpy as np
import pickle
import keras
import sys
sys.path.append("..") # Adds higher directory to python modules path.
import os
script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
up_dir = os.path.join(script_dir, '..')


models = [
    'Learn_Master/optimized_models/model_00',
    'Learn_Master/optimized_models/model_01'
]

paras_path = os.path.join(up_dir, 'Misc_Data/raw_ricoparas.pkl')
with open(paras_path, 'rb') as f:
    paras = pickle.load(f)

doc_dict_path = os.path.join(up_dir, 'Misc_Data/doc_dict.pkl')
with open(doc_dict_path, 'rb') as f:
    doc_dict = pickle.load(f)

encoded_data_paths = [
    os.path.join(up_dir, 'Word_Embeddings/Rico-Corpus/encoded_data_01.pkl'),
    os.path.join(up_dir, 'Word_Embeddings/Scientific_Articles/ricocorpus_sciart_encoded.pkl')
]
encoded_data = []
for en in encoded_data_paths:
    with open(en, 'rb') as f:
        encoded_data.append(pickle.load(f))

vocab_paths = [
    os.path.join(up_dir, 'Word_Embeddings/Rico-Corpus/ranked_vocab.pkl'),
    os.path.join(up_dir, 'Word_Embeddings/Scientific_Articles/sciart_vocab.pkl')
]
vocab = []
for v in vocab_paths:
    with open(v, 'rb') as f:
        vocab.append(pickle.load(f))

input_dims = [2, 3, 5, 10, 20, 30, 40, 50, 2]
# input_dims = [2]
output_dims = [2, 3, 4]  # greatly increases computation time

model_paths = []
for ii in range(output_dims[-1]):
    if len(range(output_dims[-1])) >= 100:
        model_paths.append(os.path.join(up_dir, 'Learn_Master/optimized_models/model_00{ii}'.format(ii=ii)))
    else:
        model_paths.append(os.path.join(up_dir, 'Learn_Master/optimized_models/model_0{ii}'.format(ii=ii)))


def fibonacci(n):
    fib = [1, 1]
    for ii in range(1, n):
        fib.append(fib[ii - 1] + fib[ii])

    fib.remove(1)  # removes first 1
    return fib


weighting_factor = fibonacci(10)
# weighting_factor = [1]
optimizers = ['sgd', 'adam', 'adagrad', 'rmsprop', 'adadelta', 'adamax']

label_paths = [
    os.path.join(up_dir, 'Ordered_Data/Full_Ordered_Labels_2Dims.pkl'),
    os.path.join(up_dir, 'Ordered_Data/Full_Ordered_Labels_3dims.pkl'),
    os.path.join(up_dir, 'Ordered_Data/Full_Ordered_Labels_4dims.pkl'),
    # os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_5dims.pkl'),
    # os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_6dims.pkl'),
    # os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_7dims.pkl'),
    # os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_8dims.pkl'),
    # os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_9dims.pkl'),
    # os.path.join(up_dir,'Ordered_Data/Full_Ordered_Labels_10Dims.pkl')
]
labels = []
for la in label_paths:
    with open(la, 'rb') as f:
        labs = pickle.load(f)

        labs = np.asarray(labs)
        labels.append(labs)

proj_paths = [
    os.path.join(up_dir, 'Misc_Data/Projection_2dims.pkl'),
    os.path.join(up_dir, 'Misc_Data/Projection_3dims.pkl'),
    os.path.join(up_dir, 'Misc_Data/Projection_4dims.pkl'),
    # os.path.join(up_dir,'Misc_Data/Projection_5dims.pkl'),
    # os.path.join(up_dir,'Misc_Data/Projection_6dims.pkl'),
    # os.path.join(up_dir,'Misc_Data/Projection_7dims.pkl'),
    # os.path.join(up_dir,'Misc_Data/Projection_8dims.pkl'),
    # os.path.join(up_dir,'Misc_Data/Projection_9dims.pkl'),
    # os.path.join(up_dir,'Misc_Data/Projection_10dims.pkl'),
]
projections = []
for pr in proj_paths:
    with open(pr, 'rb') as f:
        projections.append(pickle.load(f))

we_model_paths = [
    os.path.join(up_dir, 'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_2dims_v02'),
    os.path.join(up_dir, 'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_3dims_v02'),
    os.path.join(up_dir, 'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_5dims_v02'),
    os.path.join(up_dir, 'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_10dims_v02'),
    os.path.join(up_dir, 'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_20dims_v02'),
    os.path.join(up_dir, 'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_30dims_v02'),
    os.path.join(up_dir, 'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_40dims_v02'),
    os.path.join(up_dir, 'Word_Embeddings/Rico-Corpus/models/ricocorpus_model10000ep_50dims_v02'),
    os.path.join(up_dir, 'Word_Embeddings/Scientific_Articles/sciart_model')
    ]




with open(os.path.join(up_dir, 'predlabels.pkl'), 'rb') as f:
    prediction = pickle.load(f)
preds = prediction[0]
predlabels = prediction[1]
predlabels = predlabels[0]
# pred =

pred = np.concatenate((preds[0], preds[1], preds[2], preds[3]), axis=1)
# for ii in range(len(preds))
avgpred = np.average(pred, axis=1); avgpred = np.reshape(avgpred, (avgpred.shape[0], 1))
avgpredlabs = np.average(predlabels, axis=1); avgpredlabs = np.reshape(avgpredlabs, (avgpredlabs.shape[0], 1))

nbars = 100
y = ''
title = ''
dim = 0
colors = ['k']
# Graphing.comparative_bar_plot(avgpred, avgpredlabs, nbars, y, title, dim=dim, sort='')

graph = Graphing()
# for ii in range(10, 110, 10):
ii =150
savepath=os.path.join(up_dir, 'Figures/connected{}.pdf').format(ii)
graph.scatterplot3d(pred, predlabels, type='connect', num_points=ii, colors=colors, seed=37, savefig=savepath)
# def scatterplot3d(prediction, actual, type='', num_points=20, colors=[], styles=[], seed=24):