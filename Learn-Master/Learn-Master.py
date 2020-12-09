from atom_tag_pairing import Atom_Tag_Pairing
from atom_embedding import Atom_Embedder
from Atom_FFNN import Atom_FFNN
from Atom_RNN import Atom_RNN
from word_embedding_ricocorpus import word_embedding_ricocorpus
from word_embedding import Sciart_Word_Embedding

from Process_sciart import Process_Sciart
from Label_Text_Builder import Label_Text_Builder

# PIMS filter import


class Learn_Master():
    """

    """
    def __init__(self, neural_network, NNmodel_path='', WEmodel_path=''):
        """

        :param neural_network: Either 'ffnn' for feed-forward neural network or 'rnn' for recurrent neural network
        :param word_embedding: Either
        :param model:
        """
        self.neural_network = neural_network





    def create_tex_dataset(self):
        LTB = Label_Text_Builder()






