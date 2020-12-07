import os
import pickle

train_rnn = dict()
test_rnn = dict()

with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Atom_RNN\doc_dict.pkl', 'rb') as file:
    doc_dict = pickle.load(file)

ii = 0
for key, value in doc_dict.items():
    if ii  < 0.8*len(doc_dict):
        train_rnn[key] = value
    else:
        test_rnn[key] = value

    ii += 1




