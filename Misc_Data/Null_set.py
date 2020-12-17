import numpy as np
import pickle


with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\Projection.pkl', 'rb') as f0:
    proj = pickle.load(f0)

proj = np.asarray(proj)
proj = proj.T

with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_2dims\BOWavg_rico\test_labels.pkl', 'rb') as f1:
    labs = pickle.load(f1)



# labs = np.concatenate((labs_tr, labs_te), axis=0)

num_of_nulls = 75 # chosen because more was time prohibitive
nullset = np.empty((labs.shape))
null_arr = np.empty((labs.shape[0], labs.shape[1], num_of_nulls))
for num in range(num_of_nulls):
    for ii in range(len(nullset)):
        ri = np.random.randint(0, len(proj))
        nullset[ii, :] = proj[ri, :]
    null_arr[:, :, num] = (nullset)


with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_2dims\BOWavg_rico\nulltest_2dims.pkl', 'wb') as f2:
    pickle.dump(null_arr, f2)