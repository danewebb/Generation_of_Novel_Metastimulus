import os
import pickle
from tensorflow import keras
import json

from word_embedding_ricocorpus import word_embedding_ricocorpus
from Processing import Tex_Processing
from Process_sciart import Process_Sciart
from word_embedding import Sciart_Word_Embedding
from atom_embedding import Atom_Embedder
from atom_tag_pairing import Atom_Tag_Pairing

from Atom_RNN import Atom_RNN






atom_dict = dict()

atom_tags = []
atoms = []


with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_RNN\doc_dict.pkl', 'rb') as f1:
    doc_dict = pickle.load(f1)

with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_wordembedding\sciart_vocab.pkl', 'rb') as f2:
    vocab = pickle.load(f2)

model = keras.models.load_model(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\sciart_wordembedding\sciart_model')
weights = model.layers[0].get_weights()[0]

for key, value in doc_dict.items():

    atom_dict[key] = value
    tag = value['tag_dict']
    para = value['para_dict']
    atom_tags.append(tag['tags'])
    atoms.append(para['paragraph'])


print(len(atom_tags))
print(len(atoms))

with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\raw_ricoparas.pkl', 'wb') as f00:
    pickle.dump(atoms, f00)

with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\raw_ricotags.pkl', 'wb') as f01:
    pickle.dump(atom_tags, f01)
# TP = Tex_Processing(training_data=train_atoms, testing_data=test_atoms, vocab=vocab)
# TP.word_to_vec(vocab)
# xtrain, xtest = TP.return_encodes()

SWE = Sciart_Word_Embedding(data=atoms,
                            vocab_path=r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\sciart_vocab.pkl',
                            model_path=r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Word-Embeddings\sciart_model',
                            )


AE = Atom_Embedder(weights, vocab)
tratom_vecs = []
teatom_vecs = []

for para in xtrain:
    if para:
        tratom_vecs.append(AE.sum_atoms(para))
    else:
        # There is one empty paragraph. Error, grabbed latex code and then cleaned it.
        para = [0]
        tratom_vecs.append(AE.sum_atoms(para))

for para in xtest:
    if para:
        teatom_vecs.append(AE.sum_atoms(para))
    else:
        # There is one empty paragraph. Error, grabbed latex code and then cleaned it.
        para = [0]
        teatom_vecs.append(AE.sum_atoms(para))



with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\BOWsum_rico\train_atom_vectors_avg.pkl',
          'wb') as f4:
    pickle.dump(tratom_vecs, f4)

with open(
        r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\BOWsum_rico\test_atom_vectors_avg.pkl',
        'wb') as f5:
    pickle.dump(teatom_vecs, f5)


with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\dialectica-pimsifier\adjacency_train.json',
        'rb') as f6:
    adj = json.load(f6)

with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\Projection.pkl', 'rb') as f7:
    proj = pickle.load(f7)

ATP = Atom_Tag_Pairing(train_rnn_dict, projection=proj, adjacency=adj)
ATP.tag_pairing()
labels = ATP.projection_vectors()
with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\BOWsum_rico\train_labels.pkl', 'wb') as f8:
    pickle.dump(labels, f8)

ATP = Atom_Tag_Pairing(test_rnn_dict, projection=proj, adjacency=adj)
ATP.tag_pairing()
labels = ATP.projection_vectors()
with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\BOWsum_rico\test_labels.pkl', 'wb') as f9:
    pickle.dump(labels, f9)





