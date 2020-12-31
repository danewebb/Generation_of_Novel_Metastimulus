from matplotlib import pyplot as plt
import pickle
import numpy as np
import random


def plot_loss(graph_dict, x, y, title):
    plt.xlabel = x
    plt.ylabel = y
    plt.title = title

    for key, value in graph_dict.items():
        for kkey, vvalue in value.items():
            if kkey == 'loss':
                los = vvalue
            elif kkey == 'epochs':
                ep = vvalue


        if type(ep) != list:
            ep = range(ep)
        plt.plot(ep, los, label=key)

    plt.legend()
    plt.show()






def plot_acc(graph_dict, x, y, title):
    plt.xlabel = x
    plt.ylabel = y
    plt.title = title

    for key, value in graph_dict.items():
        for kkey, vvalue in value.items():
            if kkey == 'acc':
                acc = vvalue
            elif kkey == 'epochs':
                ep = vvalue

        if len(ep) != len(acc):
            ep = range(ep)
        plt.plot(ep, acc, label=key)

    plt.legend()
    plt.show()



def plot_pred_v_actual(prediction, actual, subx, suby, title, linestyles=[], order=[]):
    x1 = []
    y1 = []
    x2 = []
    y2 =[]

    # shapes = ['.', 'o', 'v', '>', '1', '4', 's', '+', 'X', 'd']
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'springgreen', 'tan', 'slategrey']
    colors = ['k', 'c']

    fig, ax = plt.subplots(subx, suby)
    nplots = subx*suby
    for ii in range(nplots):
        num = np.random.randint(0, len(prediction))
        x1.append(prediction[num, 0])
        y1.append(prediction[num, 1])
        x2.append(actual[num, 0])
        y2.append(actual[num, 1])


    pr = np.asarray((x1, y1))
    ac = np.asarray((x2, y2))
    tail = np.zeros((2,1))
    mm = 0
    # plt.show()

    for jj in range(subx):
        for kk in range(suby):
            ax[jj, kk].quiver((0, 0), (0, 0), ac[0, mm], ac[1, mm], scale=.05, color=colors[0])
            ax[jj, kk].quiver((0, 0), (0, 0), pr[0, mm], pr[1, mm], scale=.05, color=colors[1])
            # ax.set_xlim(-0.5, 0.5)
            # ax.set_ylim(-0.5, 0.5)

            # ax[jj, kk].quiver(*tail, (x1[mm], y1[mm]), color=colors[0], label='Prediction', linestyle=linestyles[0], zorder=order[0])
            # ax[jj, kk].quiver(*tail, (x2[mm], y2[mm]), color=colors[1], label='Actual', linestyle=linestyles[0], zorder=order[1])
            mm += 1


    # plt.legend
    fig.suptitle(title)
    plt.show()


def plot_one_loss(graph_dict, x, y, title, colors, linestyles = [], dicts_wanted=[], order=[], shift=[]):
    col_idx = 0
    plt.figure()
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.semilogy()
    col_idx = 0
    for key, value in graph_dict.items():

        if key in dicts_wanted:

            d = graph_dict[key]
            loss = d['loss'].T
            for ii in range(loss.shape[1]):
                eps = range(d['epochs'])
                if key in shift:
                    eps = eps[1:]
                    dloss = loss[:-1, ii]
                else:
                    dloss = loss[:, ii]
                plt.plot(eps, dloss, label=key, color=colors[col_idx], linestyle=linestyles[col_idx], zorder=order[col_idx])
            col_idx += 1

    plt.grid()
    # plt.legend(loc='best')#, bbox_to_anchor=(0.23, 0.62, 0.5, 0.5))
    plt.show()


def insertion_sort(sort_arr, follow_arr, sort='ascending'):
    # sorts the sort_arr. the indices sorted in the sort_arr are simultaneously changed in the follow_arr
    if len(sort_arr) != len(follow_arr):
        raise ValueError('prediction and actual arrays must be the same size.')
    for ii in range(1, len(sort_arr)):
        key = sort_arr[ii]
        fkey = follow_arr[ii]
        jj = ii-1
        if sort == 'ascending':
            while jj >= 0 and key < sort_arr[jj]:
                sort_arr[jj+1] = sort_arr[jj]
                follow_arr[jj+1] = follow_arr[jj]
                jj -= 1
        elif sort == 'descending':
            while jj >= 0 and key > sort_arr[jj]:
                sort_arr[jj + 1] = sort_arr[jj]
                follow_arr[jj + 1] = follow_arr[jj]
                jj -= 1

        sort_arr[jj+1] = key
        follow_arr[jj+1] = fkey

    return sort_arr, follow_arr


def comparative_bar_plot(prediction, actual, nbars, y, title, dim=1, width=0.35, sort='ascending', plotall=False):
    x1 = []; x2 = []

    all_idx = range(len(prediction))
    if plotall:
        ind = all_idx
        for ii in range(len(prediction)):
            x1.append(prediction[ii, dim])
            x2.append(actual[ii, dim])
    else:
        ind = np.arange(nbars)
        for ii in range(nbars):
            num = random.sample(all_idx, nbars)
            x1.append(prediction[num, dim])
            x2.append(actual[num, dim])

    if sort == 'ascending' or sort == 'descending':
        x2, x1 = insertion_sort(x2, x1, sort)


    plt.bar(ind, x1, width, label='Prediction')
    plt.bar(ind+width, x2, width, label='Actual')

    plt.ylabel(y)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()


def scatterplot3d(prediction, actual, type='', num_points=20, colors=[], styles=[], seed=24):
    random.seed(seed)
    idxs = random.sample(range(len(prediction)), num_points)

    if type == '':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cc = 0 # color counter
        for ii in idxs:

            ax.scatter3D(prediction[ii, 0], prediction[ii, 1], prediction[ii, 2], color=colors[cc])
            ax.scatter3D(actual[ii, 0], actual[ii, 1], actual[ii, 2], color=colors[cc])
            cc += 1
        plt.show()

    elif type == 'connect':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cc = 0
        for ii in idxs:
            ax.plot([prediction[ii, 0], actual[ii, 0]], [prediction[ii, 1], actual[ii, 1]], [prediction[ii, 2], actual[ii, 2]]
                    , color=colors[cc])
            ax.scatter3D(prediction[ii, 0], prediction[ii, 1], prediction[ii, 2], color='c')
            ax.scatter3D(actual[ii, 0], actual[ii, 1], actual[ii, 2], color='k')
            cc += 1
        plt.show()

    elif type == 'vector':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        count = 0
        zer = list(np.zeros(num_points))
        for ii in idxs:
            ax.plot([zer[count], prediction[ii, 0]], [zer[count], prediction[ii, 1]], [zer[count], prediction[ii, 2]],
                    linestyle=styles[0], color=colors[count])
            ax.plot([zer[count], actual[ii, 0]], [zer[count], actual[ii, 1]], [zer[count], actual[ii, 2]],
                    linestyle=styles[1], color=colors[count])
            ax.scatter3D(prediction[ii, 0], prediction[ii, 1], prediction[ii, 2], color=colors[count])
            ax.scatter3D(actual[ii, 0], actual[ii, 1], actual[ii, 2], color=colors[count])
            count += 1
        plt.show()


def multi_part(trloss, teloss, nuloss):
    tr = np.concatenate(trloss, axis=1)
    te = np.concatenate(teloss, axis=1)
    nu = np.concatenate(nuloss, axis=1)

    assert tr.shape == te.shape == nu.shape
    epochs = range(tr.size)

    plt.plot(epochs, list(tr), label='Train')
    plt.plot(epochs, te, label='Test')
    plt.plot(epochs, nu, label='Null')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dim = 0
    x = 'Epochs'
    y = 'Loss'
    title = f'RNN BOWsum 30dim-rico 3dim-out 100rnntanh-20tanh w100'
    dims = ['0-dim', '1-dim', '2-dim']
    colors = ['r', 'g', 'b']
    linestyles = ['solid', 'dashed', 'dotted']
    layer_order = [10, 5, 0]
    # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Metastimuli-Learn\Atom-FFNN\graphing_data.pkl', 'rb') as f1:
    #     prob_graph_dict = pickle.load(f1)
    #
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\results_3dims.pkl', 'rb') as f2:
        graph_dict = pickle.load(f2)

    # plot_loss(regress_graph_dict, 'Epochs', 'Loss', 'Plot1')

    # trwant = []; tewant =[]; nuwant = []
    # trloss = []; teloss = []; nuloss = []
    # for ii in range(1, 10):
    #     trdict = graph_dict[f'ff_250ep_train_rico10dims_BOWsum_w_5_3dim_relu_00_pt0{ii}']
    #     tedict = graph_dict[f'ff_250ep_test_rico10dims_BOWsum_w_5_3dim_relu_00_pt0{ii}']
    #     nudict = graph_dict[f'ff_250ep_nullset_rico10dims_BOWsum_w_5_3dim_relu_00_pt0{ii}']
    #
    #     trloss.append(trdict['loss']); teloss.append(tedict['loss']); nuloss.append(nudict['loss'])
    #
    # multi_part(trloss, teloss, nuloss)


    dicts_wanted = [
        f'rnn_100ep_train_rico_BOWsum_w_100_3dim_100rnntanh-20tanh',
        f'rnn_100ep_test_rico_BOWsum_w_100_3dim_100rnntanh-20tanh',
        # 'ff_50ep_nullset_rico10dims_ndelta_w_500_3dim_tanh_02'
                    ]
    shift = [
        f'rnn_100ep_test_rico_BOWsum_w_100_3dim_100rnntanh-20tanh',
        # 'ff_50ep_nullset_rico10dims_ndelta_w_500_3dim_tanh_02'
    ]
    plot_one_loss(graph_dict, x, y, title, colors, dicts_wanted=dicts_wanted, linestyles=linestyles, order=layer_order, shift=shift)



    # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_2dims\BOWavg_rico\prediction.pkl', 'rb') as f3:
    #     pred = pickle.load(f3)
    #


    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\BOWsum\w100\train_labels.pkl', 'rb') as f4:
        act = pickle.load(f4)
    #
    #


    # pred00_dict = graph_dict[f'rnn_100ep_pred_rico_BOWsum_w_100_3dim_100rnntanh-20tanh_00']
    # pred01_dict = graph_dict[f'rnn_100ep_pred_rico_BOWsum_w_100_3dim_100rnntanh-20tanh_01']
    # pred02_dict = graph_dict[f'rnn_100ep_pred_rico_BOWsum_w_100_3dim_100rnntanh-20tanh_02']
    # #
    # pred00 = pred00_dict['prediction']; pred00 = np.reshape(pred00, (pred00.shape[1], 1))
    # pred01 = pred01_dict['prediction']; pred01 = np.reshape(pred01, (pred01.shape[1], 1))
    # pred02 = pred02_dict['prediction']; pred02 = np.reshape(pred02, (pred02.shape[1], 1))
    # #
    # pred = np.concatenate((pred00, pred01, pred02), axis=1)

    pred_dict = graph_dict[f'rnn_100ep_pred_rico_BOWsum_w_100_3dim_100rnntanh-20tanh']
    pred = pred_dict['prediction']

    title = f'Prediction vs. Actual\n RNN BOWsum 30dim-rico w100 3dim-out 100rnntanh-20tanh'
    # subx = 3
    # suby = 3
    # line = ['solid', 'dashed']
    # ord = [5, 0]
    # plot_pred_v_actual(pred, act, subx, suby, title, linestyles=line, order=ord)


    nbars = 100
    ylab = f'Dimension 0{dim} Component'

    comparative_bar_plot(pred, act, nbars, ylab, title, dim=dim, plotall=False)

    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    scatterplot3d(pred, act, type='', colors=color_list, num_points=5, styles = ['-', 'dashed'], seed=24)