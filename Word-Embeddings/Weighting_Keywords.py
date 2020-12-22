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

        self.phrase = dict()





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
        firstword = ''
        phrase_flag = 0
        for para in self.raw_paras:
            tok_para = self.tokenize_paras(para)
            for sen in tok_para:
                for word in sen:

                    rootword = self.__lemmatize_word(word)
                    if phrase_flag == 1:
                        ph = self.phrase_check(firstword, rootword)
                        if ph != None:
                            atom_weights.append(ph['weight'])
                            phrase_flag = 0
                        else:
                            phrase_flag = 0
                    for key, val in self.keywords.items():
                        tag_words = val['words']

                        if rootword in tag_words:
                            atom_weights.append(val['weight'])
                        elif rootword == val['first'] and phrase_flag == 0:
                            firstword = val['first']
                            phrase_flag = 1
                        elif rootword in self.phrase.keys() and phrase_flag == 0:
                            firstword = rootword
                            phrase_flag = 1


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


    def phrase_check(self, firstword, secondword):
        if secondword == 'variable':
            return self.keywords['S']
        elif firstword == 'thermal':
            if secondword in self.resist:
                return self.keywords['TR']
            elif secondword in self.cap:
                return self.keywords['TC']
        elif firstword == 'fluid':
            if secondword in self.resist:
                return self.keywords['FR']
            elif secondword in self.cap:
                return self.keywords['FC']
            elif secondword in self.induct:
                return self.keywords['FL']
        elif firstword == 'angular':
            if secondword == 'velocity':
                return self.keywords['OM']
        else:
            return None




    def __init_dicts(self):
        V = dict(); I = dict(); S = dict(); R = dict()
        C = dict(); L = dict(); D = dict(); A = dict()
        T = dict(); B = dict(); f = dict(); v = dict()
        k = dict(); M = dict(); TO = dict(); OM = dict()
        J = dict(); P = dict(); Q = dict()
        FC = dict(); FR = dict(); q = dict(); TE = dict()
        TC = dict(); TR = dict(); FL = dict()
        # IN = dict()

        lump = ['inertance', 'inductance', 'inductor', 'resistance', 'resistor', 'capacitance', 'capacitor']
        self.induct = ['inertance', 'inductance', 'inductor']
        self.resist = ['resistance', 'resistor']
        self.cap = ['capacitance', 'capacitor']

        self.phrase['across'] = ['variable']
        self.phrase['through'] = ['variable']
        self.phrase['fluid'] = lump
        self.phrase['thermal'] = lump
        self.phrase['electrical'] = lump
        self.phrase['translational'] = lump
        self.phrase['rotational'] = lump
        self.phrase['angular'] = ['velocity']


        V['words'] = ['voltage', 'volt']; V['weight'] = self.wV; V['first'] = []
        v['words'] = ['velocity', 'm/s']; v['weight'] = self.wv; v['first'] = []
        OM['words'] = ['omega', 'rad/s']; OM['weight'] = self.wOM; OM['first'] = []
        P['words'] = ['pressure', 'atm', 'bar', 'psi']; P['weight'] = self.wP; P['first'] = []
        TE['words'] = ['temperature', 'temp']; TE['weight'] = self.wTE; TE['first'] = []
        S['words'] = ['source', 'across-variable', 'through-variable']; S['weight'] = self.wS; S['first'] = 'across'

        f['words'] = ['force']; f['weight'] = self.wf; f['first'] = []
        TO['words'] = ['torque', 'tau', 'nm']; TO['weight'] = self.wTO; TO['first'] = []
        I['words'] = ['current', 'amp']; I['weight'] = self.wI; I['first'] = []
        q['words'] = ['heat', 'joule', 'btu', 'watt']; q['weight'] = self.wq; q['first'] = []
        Q['words'] = ['cubic', 'volume']; Q['weight'] = self.wQ; Q['first'] = []

        A['words'] = ['a-type']; A['weight'] = self.wA; A['first'] = []
        C['words'] = ['capacitor', 'farads', 'capacitance']; C['weight'] = self.wC; C['first'] = []
        M['words'] = ['mass']; M['weight'] = self.wM; M['first'] = []
        J['words'] = ['moment', 'inertia']; J['weight'] = self.wJ; J['first'] = []
        FC['words'] = ['n/m', 'lb/m']; FC['weight'] = self.wFC; FC['first'] = 'fluid'
        TC['words'] = ['c/m', 'f/m']; TC['weight'] = self.wTC; TC['first'] = 'thermal'

        T['words'] = ['t-type']; T['weight'] = self.wT; T['first'] = []
        k['words'] = ['spring']; k['weight'] = self.wk; k['first'] = []
        L['words'] = ['inductor', 'inductance']; L['weight'] = self.wL; L['first'] = []
        FL['words'] = ['inertance']; FL['weight'] = self.wFL; FL['first'] = 'fluid'

        D['words'] = ['d-type']; D['weight'] = self.wD; D['first'] = []
        B['words'] = ['damper']; B['weight'] = self.wB; B['first'] = []
        R['words'] = ['resistor', 'resistance', 'ohm', 'impedance']; R['weight'] = self.wR; R['first'] = []
        FR['words'] = ['drag']; FR['weight'] = self.wFR; FR['first'] = 'fluid'
        TR['words'] = ['k/w']; TR['weight'] = self.wTR; TR['first'] = 'thermal'



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
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_10dims\ndelta_rico\all_atoms.pkl', 'rb') as f4:
        atoms = pickle.load(f4)



    indim = 10
    new_atoms = np.empty((len(atoms), indim))
    for ii, ele in enumerate(atoms):
        if ele == []:
            nele = np.zeros((1, indim))
            new_atoms[ii, :] = nele
        else:
            new_atoms[ii, :] = np.asarray(ele)



    WK = Weighting_Keyword(raw_paras,
                           wV=5.1, wI=7.1, wS=10.9, wR=100.1, wC=300.1, wL=50.1, wD=500.9, wA=290.9, wT=65.9,
                           wB=111.3, wf=3.3, wv=4.3, wk=57.3, wM=290.3, wTO=8.5,
                           wOM=9.5, wJ=322.5, wP=6.7, wQ=8.7, wFC=344.7, wFR=131.7, wq=11.8, wTE=6.8,
                           wTC=366.8, wTR=88.8, wFL=37.7
                           )
    WK.keyword_search()
    weighted_ordered_atoms = WK.apply_weights(new_atoms)


    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_10dims\ndelta_rico\weighted_all_atoms_5_500.pkl', 'wb') as f5:
        pickle.dump(weighted_ordered_atoms, f5)
