import os
import numpy as np
import pickle
import json
import re
import math
import itertools
from sklearn.neighbors import NearestNeighbors
from scipy import spatial

class Atom_Tag_Pairing():
    def __init__(self, data_dict, adjacency=None, projection=None, prediction=None):

        self.data = data_dict
        self.adj = adjacency
        self.proj = projection

        self.pred = prediction # output vector from the NN

        self.tagnum = [] # stores adjacency tag groups.
        self.va = [] # adjacency values
        self.vd = [] # data values

        if prediction:
            self.pred_proj = np.zeros((self.pred.shape))

        self.proj_idx = []
    def tag_pairing(self):



        pattern = r'\w+'
        for v1 in self.data.values():
            val1 = v1['tag_dict']
            self.vd.append(val1['tags'])

        for val2 in self.adj['nodes'].keys():
            strip = re.findall(pattern, val2)
            self.va.append(strip)



        # grab index of matching tag intersection in va for each vd tag
        # data_tag_idxs = [idx for idx, tag in enumerate(vd) if tag in set(va)]


        # set_va = set(tuple(row) for row in va) # need to turn list of list into set of tuples otherwise error
        # set_vd = set(tuple(row) for row in vd)
        for tag_group in self.vd:
            # grabs the index of the tag group from the adjacency list
            # if tag_group in va:
            #     tagnum.append(va.index(set_tag))
            if tag_group != ['']:
                if any(tag_group.count(ele) > 1 for ele in tag_group):
                    tag_group = self.__lazy_tag_fix(tag_group)
                order_list = self.__every_order(tag_group)
                if any(item in order_list for item in self.va):
                    # if any of the orientations in order_list match tag group in va
                    res = [ii for ii, val in enumerate(order_list) if val in self.va]
                    if len(res) > 1:
                        ValueError('Multiple orientations of same tag group in va')

                    resint = int(res[0])
                    self.tagnum.append(self.va.index(order_list[resint]))
                else:
                    print(f'Tag without va pair {tag_group}')
                    self.tagnum.append(len(self.va))
            else:
                self.tagnum.append(len(self.va))

        lenva = len(self.va)
        lenvd = len(self.vd)
        lentagnum = len(self.tagnum)



    def projection_vectors(self):
        # naive way
        # array of vectors
        # Projection method
        lenva = len(self.va)
        tagvec = []
        for num in self.tagnum:
            if num == lenva:
                tagvec.append(np.zeros([2, 1]))
            else:
                temp = self.proj[:, num]
                temp = temp.reshape((temp.shape[0], 1))
                tagvec.append(temp)


        labels = np.concatenate(tagvec, axis=1)

        labels = labels.T

        return labels



    def nearest_neighbor(self):
        # nearest neighbor search between the output vectors from the ANN and the projection matrix
        lenvec = len(self.pred)
        proj_idx = [] # list of indices of the projection vectors in self.proj that are the nearest neighbor to the current vec of self.out_vec

        nbrs = NearestNeighbors(n_neighbors=1).fit(self.proj)
        for ii, vec in enumerate(self.pred):
            vec = vec.reshape(1, -1)
            kn = nbrs.kneighbors(vec)
            proj_idx = int(kn[1]) # kn[1] is the index of self.proj that vec is nearest
            self.pred_proj[ii, :] = self.proj[proj_idx]
            self.proj_idx.append(proj_idx)

        return self.pred_proj


    def proj_to_nodes(self):
        adj_nodes = self.adj['nodes']
        pred_nodes = []
        try:
            for idx in self.proj_idx:
                pred_nodes.append(adj_nodes.values[str(idx)])
        except:
            rev_adj_nodes = dict()
            for key, val in adj_nodes.items():
                rev_adj_nodes[val] = key

            for idx in self.proj_idx:
                pred_nodes.append(rev_adj_nodes[idx])

        return pred_nodes


    # def proj_accuracy(self):
    # def node_accuracy(self):


    def one_hotify(self):
        # One-hot method for adjacency nodes
        # assume undefined tag groups are defined as len(tagnum)
        rows = len(self.tagnum)
        cols =  max(self.tagnum)
        one_hot = np.zeros((rows, cols-1), dtype=np.int32) # -1 because we don't want throwaway values in one-hot
        for row, tag in enumerate(self.tagnum):
            if tag != cols:
                one_hot[row, tag] = 1

        return one_hot


    def __every_order(self, tag_group):
        len_group = len(tag_group)
        # orientations_num = math.factorial(len_group) # number of possible combinations
        poss_lists = list(itertools.permutations(tag_group, len_group))
        listify = []
        for jj in poss_lists:
            listify.append(list(jj))
        return listify

    def __lazy_tag_fix(self, tag_group):
        # some tag groups accidentally have doubled tags. This function removes duplicates
        # fixes errors with initial tagging
        res = []
        for tag in tag_group:
            if tag not in res:
                res.append(tag)
        return res



if __name__ == '__main__':
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\doc_dict.pkl', 'rb') as f1:
        data = pickle.load(f1)

    with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\adjacency_train.json', 'rb') as f2:
        adj = json.load(f2)

    with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\Projection.pkl', 'rb') as f3:
        proj = pickle.load(f3)


    # Training labels
    ATP = Atom_Tag_Pairing(data, adjacency=adj, projection=proj)

    ATP.tag_pairing()
    # labels = ATP.one_hotify()
    labels = ATP.projection_vectors()
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Full_Ordered_Labels.pkl', 'wb') as f4:
        pickle.dump(labels, f4)

    # Inverse

    # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\prediction.pkl', 'rb') as f5:
    #     pred = pickle.load(f5)
    # ATP = Atom_Tag_Pairing(data, adjacency=adj, projection=proj.T, prediction=pred)
    # nearest_projections = ATP.nearest_neighbor()
    # predicted_nodes = ATP.proj_to_nodes()
    #
    # print('hello world')



