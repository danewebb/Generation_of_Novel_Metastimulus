import pickle
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer




class Weighting_Keyword:

    def __init__(self, vocab, weight, tags=None):

        self.tags = tags
        self.paras = None
        self.keywords = dict()

        self.vocab = vocab
        self.w = weight

        # wIN = 1

        self.phrase = dict()

        self.enc_keywords = dict()



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


    def __flatlist(self, lst):
        flat_list = []
        flat_list = ' '.join(lst)
        return flat_list

    def keyword_search(self, para):

        firstword = ''
        phrase_flag = 0

        atom_weights = np.ones((len(para)))
        for idx, enc in enumerate(para):

            for key, val in self.keywords.items():
                tag_encodes = val['enc']
                if enc in tag_encodes:
                    atom_weights[idx] = val['weight']


        # for idx, word in enumerate(para):
        #     if phrase_flag == 1:
        #         ph = self.phrase_check(firstword, rootword)
        #         if ph != None:
        #             atom_weights[idx] = ph['weight']
        #             phrase_flag = 0
        #         else:
        #             phrase_flag = 0
        #     for key, val in self.keywords.items():
        #         tag_words = val['words']
        #
        #         if rootword in tag_words:
        #             atom_weights[idx] = val['weight']
        #         elif rootword == val['first'] and phrase_flag == 0:
        #             firstword = val['first']
        #             phrase_flag = 1
        #         elif rootword in self.phrase.keys() and phrase_flag == 0:
        #             firstword = rootword
        #             phrase_flag = 1




        return atom_weights


    def keywords_in_vocab(self):
        vals = self.keywords.values()
        firsts = self.phrase.keys()
        seconds = self.phrase.values()

        for idx, enc in enumerate(self.vocab):
            for node, node_dict in self.keywords.items():
                for word in node_dict['words']:
                    if enc == word:
                        node_dict['enc'].append(idx)
                for word in node_dict['first']:
                    if enc == word:
                        node_dict['first_enc'].append(idx)

            # for key, val in self.phrase:
            #     if key == enc:
            #         self.phrase



    def apply_weights(self, embedded_atom, atom_weights):
        atoms = np.asarray(embedded_atom)
        weighted_atom = np.ones(atoms.shape)
        for num, atom in enumerate(atoms):
            weight = atom_weights[num]
            weighted_atom[num, :] = weight*atom

        return weighted_atom


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
        self.induct = ['inertance', 'inductance', 'inductor', 'inductors']
        self.resist = ['resistance', 'resistor', 'resistors']
        self.cap = ['capacitance', 'capacitor', 'capacitors']

        self.phrase['across'] = ['variable']
        self.phrase['through'] = ['variable']
        self.phrase['fluid'] = lump
        self.phrase['thermal'] = lump
        self.phrase['electrical'] = lump
        self.phrase['translational'] = lump
        self.phrase['rotational'] = lump
        self.phrase['angular'] = ['velocity']


        V['words'] = ['voltage', 'volt', 'volts']; V['weight'] = self.w; V['first'] = []
        v['words'] = ['velocity', 'm/s']; v['weight'] = self.w; v['first'] = []
        OM['words'] = ['omega', 'rad/s']; OM['weight'] = self.w; OM['first'] = []
        P['words'] = ['pressure', 'atm', 'bar', 'psi']; P['weight'] = self.w; P['first'] = []
        TE['words'] = ['temperature', 'temperatures', 'temp']; TE['weight'] = self.w; TE['first'] = []
        S['words'] = ['source', 'sources', 'across-variable', 'through-variable']; S['weight'] = self.w; S['first'] = ['across']

        f['words'] = ['force', 'forces']; f['weight'] = self.w; f['first'] = []
        TO['words'] = ['torque', 'tau', 'nm', 'torques']; TO['weight'] = self.w; TO['first'] = []
        I['words'] = ['current', 'amp', 'amps']; I['weight'] = self.w; I['first'] = []
        q['words'] = ['heat', 'joule', 'btu', 'watt', 'watts']; q['weight'] = self.w; q['first'] = []
        Q['words'] = ['cubic', 'volume']; Q['weight'] = self.w; Q['first'] = []

        A['words'] = ['a-type']; A['weight'] = self.w; A['first'] = []
        C['words'] = ['capacitor', 'farads', 'capacitance', 'capacitors']; C['weight'] = self.w; C['first'] = []
        M['words'] = ['mass', 'masses']; M['weight'] = self.w; M['first'] = []
        J['words'] = ['moment', 'inertia', 'moments']; J['weight'] = self.w; J['first'] = []
        FC['words'] = ['n/m', 'lb/m']; FC['weight'] = self.w; FC['first'] = ['fluid', 'fluids']
        TC['words'] = ['c/m', 'f/m']; TC['weight'] = self.w; TC['first'] = ['thermal', 'thermals']

        T['words'] = ['t-type']; T['weight'] = self.w; T['first'] = []
        k['words'] = ['spring', 'springs']; k['weight'] = self.w; k['first'] = []
        L['words'] = ['inductor', 'inductance', 'inductors']; L['weight'] = self.w; L['first'] = []
        FL['words'] = ['inertance']; FL['weight'] = self.w; FL['first'] = ['fluid', 'fluids']

        D['words'] = ['d-type']; D['weight'] = self.w; D['first'] = []
        B['words'] = ['damper']; B['weight'] = self.w; B['first'] = []
        R['words'] = ['resistor', 'resistors', 'resistance', 'ohm', 'impedance']; R['weight'] = self.w; R['first'] = []
        FR['words'] = ['drag']; FR['weight'] = self.w; FR['first'] = ['fluid', 'fluids']
        TR['words'] = ['k/w']; TR['weight'] = self.w; TR['first'] = ['thermal', 'thermals']



        self.keywords['V'] = V; self.keywords['I'] = I; self.keywords['S'] = S; self.keywords['R'] = R
        self.keywords['C'] = C; self.keywords['L'] = L;self. keywords['D'] = D; self.keywords['A'] = A
        self.keywords['T'] = T; self.keywords['B'] = B; self.keywords['f'] = f; self.keywords['v'] = v
        self.keywords['k'] = k; self.keywords['M'] = M; self.keywords['TO'] = TO; self.keywords['OM'] = OM
        self.keywords['J'] = J; self.keywords['P'] = P; self.keywords['Q'] = Q
        self.keywords['FC'] = FC; self.keywords['FR'] = FR; self.keywords['q'] = q; self.keywords['TE'] = TE
        self.keywords['TC'] = TC; self.keywords['TR'] = TR; self.keywords['FL'] = FL
        # self.keywords['IN'] = IN

        for key, val in self.keywords.items():
            val['enc'] = []
            val['first_enc'] = []

if __name__ == '__main__':
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\raw_ricoparas.pkl', 'rb') as f2:
        raw_paras = pickle.load(f2)
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_10dims\ndelta_rico\all_atoms.pkl', 'rb') as f4:
        atoms = pickle.load(f4)


    weight = 5
    indim = 10
    new_atoms = np.empty((len(atoms), indim))
    for ii, ele in enumerate(atoms):
        if ele == []:
            nele = np.zeros((1, indim))
            new_atoms[ii, :] = nele
        else:
            new_atoms[ii, :] = np.asarray(ele)



    WK = Weighting_Keyword(raw_paras, weight,
                           )
    WK.keyword_search()
    weighted_ordered_atoms = WK.apply_weights(new_atoms)


    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_10dims\ndelta_rico\weighted_all_atoms_5_500.pkl', 'wb') as f5:
        pickle.dump(weighted_ordered_atoms, f5)
