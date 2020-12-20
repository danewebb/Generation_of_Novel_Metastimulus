import pickle
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer




class Weighting_Keyword:

    def __init__(self, paras, tags=None,
                 wV=1.0, wI=1.0, wS=1.0, wR=1.0, wC=1.0, wL=1.0, wD=1.0,
                 wA=1.0, wT=1.0, wB=1.0, wf=1.0, wv=1.0, wk=1.0, wM=1.0,
                 wTO=1.0, wOM=1.0, wJ=1.0, wP=1.0, wQ=1.0, wFC=1.0,
                 wFR=1.0, wq=1.0, wTE=1.0, wTC=1.0, wTR=1.0, wFL=1.0
                 ):
        self.raw_paras = paras
        self.tags = tags
        self.paras = None
        self.keywords = dict()

        self.all_weights = []

        self.wV = wV; self.wI = wI; self.wS = wS; self.wR = wR
        self.wC = wC; self.wL = wL; self.wD = wD; self.wA = wA
        self.wT = wT; self.wB = wB; self.wf = wf; self.wv = wv
        self.wk = wk; self.wM = wM; self.wTO = wTO; self.wOM = wOM
        self.wJ = wJ; self.wP = wP; self.wQ = wQ
        self.wFC = wFC; self.wFR = wFR; self.wq = wq; self.wTE = wTE
        self.wTC = wTC; self.wTR = wTR; self.wFL = wFL

        # wIN = 1


        self.__init_dicts()


    def list_unique_tags(self):
        # creates a list of the unique tags and creates tag dictionaries
        unique_tags = []
        for tag_group in self.tags:
            for self.tag in tag_group:
                if self.tag not in unique_tags:
                    unique_tags.append(self.tag)


        return unique_tags

    def tokenize_paras(self, para):
        tokenized = []
        sentences = []

        for ele in para:
            plower = ele.lower()
            sentences.append(nltk.word_tokenize(plower))

        return sentences

    def __lemmatize_word(self, word):
        lemmatizer = WordNetLemmatizer()
        lword = lemmatizer.lemmatize(word)
        return lword

    def keyword_search(self):

        atom_weights = []
        for para in self.raw_paras:
            tok_para = self.tokenize_paras(para)
            for sen in tok_para:
                for word in sen:
                    rootword = self.__lemmatize_word(word)
                    for key, val in self.keywords.items():
                        tag_words = val['words']
                        if rootword in tag_words:
                            atom_weights.append(val['weight'])
            para_weights = sum(atom_weights)
            if para_weights == 0:
                para_weights = 1
            self.all_weights.append(para_weights)
            atom_weights = []


    def apply_weights(self, embedded_atoms):
        atoms = np.asarray(embedded_atoms)
        weighted_atoms = np.ones(atoms.shape)
        for num, atom in enumerate(atoms):
            weight = self.all_weights[num]
            weighted_atoms[num, :] = weight*atom

        return weighted_atoms

    def __init_dicts(self):
        V = dict(); I = dict(); S = dict(); R = dict()
        C = dict(); L = dict(); D = dict(); A = dict()
        T = dict(); B = dict(); f = dict(); v = dict()
        k = dict(); M = dict(); TO = dict(); OM = dict()
        J = dict(); P = dict(); Q = dict()
        FC = dict(); FR = dict(); q = dict(); TE = dict()
        TC = dict(); TR = dict(); FL = dict()
        # IN = dict()

        V['words'] = ['voltage', 'volt']; V['weight'] = self.wV
        v['words'] = ['velocity', 'm/s']; v['weight'] = self.wv
        OM['words'] = ['angular', 'velocity', 'omega', 'rad/s']; OM['weight'] = self.wOM #
        P['words'] = ['pressure', 'atm', 'bar', 'psi']; P['weight'] = self.wP
        TE['words'] = ['temperature', 'temp']; TE['weight'] = self.wTE#
        S['words'] = ['source']; S['weight'] = self.wS

        f['words'] = ['force']; f['weight'] = self.wf
        TO['words'] = ['torque', 'tau', 'nm']; TO['weight'] = self.wTO
        I['words'] = ['current', 'amp']; I['weight'] = self.wI
        q['words'] = ['heat', 'joule', 'btu', 'watt']; q['weight'] = self.wq
        Q['words'] = ['cubic', 'volume']; Q['weight'] = self.wQ

        A['words'] = ['a-type']; A['weight'] = self.wA
        C['words'] = ['capacitor', 'farads', 'capacitance']; C['weight'] = self.wC
        M['words'] = ['mass']; M['weight'] = self.wM
        J['words'] = ['moment', 'inertia']; J['weight'] = self.wJ
        FC['words'] = ['n/m', 'lb/m']; FC['weight'] = self.wFC #
        TC['words'] = ['c/m', 'f/m']; TC['weight'] = self.wTC #

        T['words'] = ['t-type']; T['weight'] = self.wT
        k['words'] = ['spring']; k['weight'] = self.wk
        L['words'] = ['inductor', 'inductance']; L['weight'] = self.wL
        FL['words'] = ['inertance']; FL['weight'] = self.wFL # fluid inductor

        D['words'] = ['d-type']; D['weight'] = self.wD
        B['words'] = ['damper']; B['weight'] = self.wB
        R['words'] = ['resistor', 'resistance', 'ohm', 'impedance']; R['weight'] = self.wR
        FR['words'] = ['drag']; FR['weight'] = self.wFR
        TR['words'] = ['k/w']; TR['weight'] = self.wTR



        self.keywords['V'] = V; self.keywords['I'] = I; self.keywords['S'] = S; self.keywords['R'] = R
        self.keywords['C'] = C; self.keywords['L'] = L;self. keywords['D'] = D; self.keywords['A'] = A
        self.keywords['T'] = T; self.keywords['B'] = B; self.keywords['f'] = f; self.keywords['v'] = v
        self.keywords['k'] = k; self.keywords['M'] = M; self.keywords['TO'] = TO; self.keywords['OM'] = OM
        self.keywords['J'] = J; self.keywords['P'] = P; self.keywords['Q'] = Q
        self.keywords['FC'] = FC; self.keywords['FR'] = FR; self.keywords['q'] = q; self.keywords['TE'] = TE
        self.keywords['TC'] = TC; self.keywords['TR'] = TR; self.keywords['FL'] = FL
        # self.keywords['IN'] = IN





if __name__ == '__main__':
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\raw_ricoparas.pkl', 'rb') as f2:
        raw_paras = pickle.load(f2)
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_10dims\BOWavg_rico\all_atoms.pkl', 'rb') as f4:
        atoms = pickle.load(f4)





    WK = Weighting_Keyword(raw_paras,
                           wV=1.1, wI=1.1, wS=1.9, wR=1.1, wC=1.1, wL=1.1, wD=1.9, wA=1.9, wT=1.9,
                           wB=1.3, wf=1.3, wv=1.3, wk=1.3, wM=1.3, wTO=1.5,
                           wOM=1.5, wJ=1.5, wP=1.7, wQ=1.7, wFC=1.7, wFR=1.7, wq=1.8, wTE=1.8,
                           wTC=1.8, wTR=1.8, wFL=1.7
                           )
    WK.keyword_search()
    weighted_ordered_atoms = WK.apply_weights(atoms)


    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_10dims\BOWavg_rico\weighted_all_atoms1.pkl', 'wb') as f5:
        pickle.dump(weighted_ordered_atoms, f5)
