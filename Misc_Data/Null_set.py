import numpy as np
import pickle


with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\Projection.pkl', 'rb') as f0:
    proj = pickle.load(f0)

proj = np.asarray(proj)
proj = proj.T

with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\BOWsum_rico\test_labels.pkl', 'rb') as f1:
    labs_te = pickle.load(f1)

with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\BOWsum_rico\train_labels.pkl', 'rb') as f2:
    labs_tr = pickle.load(f2)

labs = np.concatenate((labs_tr, labs_te), axis=0)

num_of_nulls = 25 # chosen because more was time prohibitive
nullset = np.empty((labs.shape))
null_list = []
for num in range(num_of_nulls):
    for ii in range(len(nullset)):
        ri = np.random.randint(0, len(proj))
        nullset[ii, :] = proj[ri, :]
    null_list.append(nullset)


with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\nullfull_list_of_sets.pkl', 'wb') as f2:
    pickle.dump(null_list, f2)