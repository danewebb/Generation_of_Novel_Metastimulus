import pickle
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer




class Weighting_Keyword:

    def __init__(self, paras, tags):
        self.paras = paras
        self.tags = tags

        self.keywords = dict()

    def list_unique_tags(self):
        # creates a list of the unique tags and creates tag dictionaries
        unique_tags = []
        for tag_group in self.tags:
            for self.tag in tag_group:
                if self.tag not in unique_tags:
                    unique_tags.append(self.tag)


        return unique_tags

    def tokenize_paras(self):
        tokenized = []
        sentences = []
        for para in self.paras:
            for ele in para:
                plower = ele.lower()
                sentences.append(nltk.word_tokenize(plower))
            tokenized.append(sentences)

        return tokenized

    def __lemmatize_word(self, word):
        lemmatizer = WordNetLemmatizer()
        lword = lemmatizer.lemmatize(word)
        return lword

    def keyword_search(self):
        all_weights = []
        atom_weights = []
        for para, tag_group in zip(self.paras, self.tags):
            for word in para:
                rootword = self.__lemmatize_word(word)
                for key, val in self.keywords.items():
                    tag_words = val['words']
                    if rootword in tag_words:
                        atom_weights.append(val['weight'])



    def __init_dicts(self):
        V = dict(); I = dict(); S = dict(); R = dict()
        C = dict(); L = dict(); D = dict(); A = dict()
        T = dict(); B = dict(); f = dict(); v = dict()
        k = dict(); M = dict(); TO = dict(); OM = dict()
        J = dict(); P = dict(); Q = dict(); IN = dict()
        FC = dict(); FR = dict(); q = dict(); TE = dict()
        TC = dict(); TR = dict(); FL = dict()

        V['words'] = ['voltage', 'volt']
        v['words'] = ['velocity', 'm/s']
        OM['words'] = ['angular', 'velocity', 'omega', 'rad/s'] #
        P['words'] = ['pressure', 'atm', 'bar', 'psi']
        TE['words'] = ['temperature', 'temp'] #
        S['words'] = ['source']

        f['words'] = ['force']
        TO['words'] = ['torque', 'tau', 'nm']
        I['words'] = ['current', 'amp']
        q['words'] = ['heat', 'joule', 'btu', 'watt']
        Q['words'] = ['cubic', 'volume']

        A['words'] = ['a-type']
        C['words'] = ['capacitor', 'farads', 'capacitance']
        M['words'] = ['mass']
        J['words'] = ['moment', 'inertia']
        FC['words'] = ['fluid'] #
        TC['words'] = ['thermal'] #

        T['words'] = ['t-type']
        k['words'] = ['spring']
        # rotational spring???
        L['words'] = ['inductor', 'inductance']
        FL['words'] = ['inertance'] # fluid inductor

        D['words'] = ['d-type']
        B['words'] = ['damper']
        # rotational damper
        R['words'] = ['resistor', 'resistance', 'ohm']
        # fluid resistor



        self.keywords['V'] = V; self.keywords['I'] = I; self.keywords['S'] = S; self.keywords['R'] = R
        self.keywords['C'] = C; self.keywords['L'] = L;self. keywords['D'] = D; self.keywords['A'] = A
        self.keywords['T'] = T; self.keywords['B'] = B; self.keywords['f'] = f; self.keywords['v'] = v
        self.keywords['k'] = k; self.keywords['M'] = M; self.keywords['TO'] = TO; self.keywords['OM'] = OM
        self.keywords['J'] = J; self.keywords['P'] = P; self.keywords['Q'] = Q; self.keywords['IN'] = IN
        self.keywords['FC'] = FC; self.keywords['FR'] = FR; self.keywords['q'] = q; self.keywords['TE'] = TE
        self.keywords['TC'] = TC; self.keywords['TR'] = TR; self.keywords['FL'] = FL






    def hardcode(self):
        wV = 1; wI = 1; wS = 1; wR = 1
        wC = 1; wL = 1; wD = 1; wA = 1
        wT = 1; wB = 1; wf = 1; wv = 1
        wk = 1; wM = 1; wTO = 1; wOM = 1
        wJ = 1; wP = 1; wQ = 1; wIN = 1
        wFC = 1; wFR = 1; wq = 1; wTE = 1
        wTC = 1; wTR = 1; wFL = 1


if __name__ == '__main__':
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\doc_dict.pkl', 'rb') as f1:
        doc = pickle.load(f1)
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\raw_ricoparas.pkl', 'rb') as f2:
        raw_paras = pickle.load(f2)
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\raw_ricotags.pkl', 'rb') as f3:
        tags =  pickle.load(f3)
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_2dims\BOWavg_rico\all_atoms.pkl', 'rb') as f4:
        atoms = pickle.load(f4)
    for num, para in enumerate(raw_paras):
        print(para)
        print(tags[num])
        print('-----------------------------------------------------------------')
        print('-----------------------------------------------------------------\n')


    # WK = Weighting_Keyword(raw_paras, tags)
    # unique_tags = WK.list_unique_tags()
    # token_paras = WK.tokenize_paras()

