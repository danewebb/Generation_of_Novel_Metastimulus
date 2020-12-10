import os
import pickle
from tensorflow import keras
import json

from word_embedding_ricocorpus import word_embedding_ricocorpus
from Processing import Tex_Processing
from atom_embedding import Atom_Embedder
from atom_tag_pairing import Atom_Tag_Pairing

from Atom_RNN import Atom_RNN






train_rnn_dict = dict()
train_tags = []
train_atoms = []

test_rnn_dict = dict()
test_tags = []
test_atoms = []

with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_RNN\doc_dict.pkl', 'rb') as f1:
    doc_dict = pickle.load(f1)

with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\ricocorpus_wordembedding\ranked_vocab.pkl', 'rb') as f2:
    vocab = pickle.load(f2)

model = keras.models.load_model(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\ricocorpus_model')
weights = model.layers[0].get_weights()[0]

ii = 0
for key, value in doc_dict.items():
    if ii  < 0.8*len(doc_dict):
        train_rnn_dict[key] = value
        tag_dict = value['tag_dict']
        para_dict = value['para_dict']
        train_tags.append(tag_dict['tags'])
        train_atoms.append(para_dict['paragraph'])

    else:
        test_rnn_dict[key] = value
        tag_dict = value['tag_dict']
        para_dict = value['para_dict']
        test_tags.append(tag_dict['tags'])
        test_atoms.append(para_dict['paragraph'])
    ii += 1





TP = Tex_Processing(training_data=train_atoms, testing_data=test_atoms, vocab=vocab)
TP.word_to_vec(vocab)
xtrain, xtest = TP.return_encodes()




AE = Atom_Embedder(weights, vocab)
tratom_vecs = []
teatom_vecs = []

for para in xtrain:
    if para:
        tratom_vecs.append(AE.avg_atoms(para))
    else:
        # There is one empty paragraph. Error, grabbed latex code and then cleaned it.
        para = [0]
        tratom_vecs.append(AE.avg_atoms(para))

for para in xtest:
    if para:
        teatom_vecs.append(AE.avg_atoms(para))
    else:
        # There is one empty paragraph. Error, grabbed latex code and then cleaned it.
        para = [0]
        teatom_vecs.append(AE.avg_atoms(para))


# with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\RNN_Data\train_atom_vectors_avg_rnn.pkl',
#           'wb') as f4:
#     pickle.dump(tratom_vecs, f4)
#
# with open(
#         r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\RNN_Data\test_atom_vectors_avg_rnn.pkl',
#         'wb') as f5:
#     pickle.dump(teatom_vecs, f5)


with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\dialectica-pimsifier\adjacency_train.json',
        'rb') as f6:
    adj = json.load(f6)

with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\Projection.pkl', 'rb') as f7:
    proj = pickle.load(f7)

ATP = Atom_Tag_Pairing(test_rnn_dict, projection=proj, adjacency=adj)
ATP.tag_pairing()
labels = ATP.projection_vectors()
with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\RNN_Data\test_labels_rnn.pkl', 'wb') as f8:
    pickle.dump(labels, f8)

