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
    if sort_arr.shape[0] != follow_arr.shape[0]:
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
                follow_arr[jj + 1, :] = follow_arr[jj]
                jj -= 1

        sort_arr[jj+1] = key
        follow_arr[jj+1] = fkey

    return sort_arr, follow_arr



def ordered_components(prediction, actual, num, color_list):
    all_idx = range(actual.shape[0])
    dims = actual.shape[1]
    # nump = random.sample(all_idx, num)
    sorter = actual [:, 0]
    a = actual[:, 1:]
    follower = np.concatenate((a, prediction), axis=1)
    sor, fol = insertion_sort(sorter, follower)
    sor = np.reshape(sor, (sor.shape[0], 1))
    act = np.concatenate((sor, fol[:, :2]), axis=1)
    pred = fol[:, actual.shape[1]-1:]
    for ii in range(dims):

        plt.plot(all_idx, act[:, ii], color=color_list[ii], label=f'Dim{ii} Actual')
        plt.plot(all_idx, pred[:, ii], linestyle='dashed', color=color_list[ii], label=f'Dim{ii} Prediction')


    plt.legend()
    plt.show()

def comparative_bar_plot(prediction, actual, nbars, y, title, dim=1, width=0.35, sort='ascending', plotall=False):
    # x1 = []; x2 = []
    # todone Rico save routine

    all_idx = range(len(prediction))
    if plotall:
        ind = all_idx
        # for ii in range(len(prediction)):
        x1 = prediction[:, dim]
        x2 = actual[:, dim]
    else:
        ind = np.arange(nbars)
        num = random.sample(all_idx, nbars)

        x1 = np.empty((nbars, ))
        x2 = np.empty((nbars, ))
        for ii in range(nbars):
            x1[ii] = prediction[num[ii], dim]
            x2[ii] = actual[num[ii], dim]

    if sort == 'ascending' or sort == 'descending':
        x2, x1 = insertion_sort(x2, x1, sort)

    plt.figure(figsize=(6.4,2.4))

    plt.bar(ind, x1, width, label='prediction')
    plt.bar(ind+width, x2, width, label='actual')

    plt.ylabel(y)
    plt.title(title)
    plt.legend(loc='best', fontsize=7)
    plt.ylabel(y, fontsize=9)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.grid()
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xticks([50], visible=True, rotation="horizontal")
    savefig = 'D:/Documents/THESIS/Figures/HCI_2021/CBP_dim{}.pdf'.format(dim)
    plt.savefig(savefig, bbox_inches='tight', format='pdf')



def scatterplot3d(prediction, actual, type='', num_points=20, colors=[], styles=[], seed=24):
    # todone Rico save routine
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
            ax.scatter3D(actual[ii, 0], actual[ii, 1], actual[ii, 2], color=colors[count], marker='x')
            count += 1

    else:
        raise Exception('unknown argument for type')

    plt.autoscale(enable=True, axis='x', tight=True)
    savefig = 'D:/Documents/THESIS/Figures/HCI_2021/vector.pdf'.format(dim)
    plt.savefig(savefig, bbox_inches='tight', format='pdf')


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


def plot_metaloss(train, test, title, colors, x, y, null=None, linestyles = None, order=None, dim=0, merge=False):
    ep = range(train.shape[1])
    shift = range(1, len(ep)+1)

    fig, ax = plt.subplots()
    plt.figure(figsize=(6.4,2.4))
    tr = train; te = test; nu = null

        # tr = train[ii]; te = test[ii]; nu = null[ii]
        # nu = np.average(nu, axis=2)
        # title = title.format(dim=dim, ii=ii)
    # try:
    if linestyles is None:
        plt.semilogy(ep, tr[dim, :], label='Train')
        plt.semilogy(shift, te[dim, :], label='Test')
        plt.semilogy(shift, nu[dim, :], label='Nullset')
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        if not merge:
            plt.show()
            return
    else:
        plt.semilogy(ep, tr[dim, :], color=colors[0], linestyle=linestyles[0], zorder=order[0], label='Train')
        plt.semilogy(shift, te[dim, :], color=colors[1], linestyle=linestyles[1], zorder=order[1], label='Test')
        plt.semilogy(shift, nu[dim, :], color=colors[2], linestyle=linestyles[2], zorder=order[2], label='Nullset')
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        if not merge:
            # ax.tick_params(labelbottom=False)
            tic = []

            plt.xticks([200], visible=True, rotation="horizontal")

            plt.autoscale(enable=True, axis='x', tight=True)

            savepath = 'D:/Documents/THESIS/Figures/HCI_2021/metaloss_dim{}.pdf'.format(dim)
            plt.savefig(savepath, bbox_inches='tight', format='pdf')
            return
    # except:
    #     print('More dimensions than available for some of the data')
    #     if linestyles is None:
    #         plt.semilogy(ep, tr[-1, :], label='Train')
    #         plt.semilogy(shift, te[-1, :], label='Test')
    #         plt.semilogy(shift, nu[-1, :], label='Nullset')
    #         plt.title(title)
    #         plt.xlabel('Epochs')
    #         plt.ylabel('Loss')
    #         plt.legend()
    #         if not merge:
    #             plt.show()
    #             return
    #     else:
    #         plt.semilogy(ep, tr[-1, :], color=colors[0], linestyle=linestyles[0], zorder=order[0], label='Train')
    #         plt.semilogy(shift, te[-1, :], color=colors[1], linestyle=linestyles[1], zorder=order[1], label='Test')
    #         plt.semilogy(shift, nu[-1, :], color=colors[2], linestyle=linestyles[2], zorder=order[2], label='Nullset')
    #         plt.title(title)
    #         plt.xlabel(x)
    #         plt.ylabel(y)
    #         plt.legend()
    #         if not merge:
    #             plt.show()
    #             return



def plot_final_loss(train, test, title, x, y, colors, bestcount, ylim=None, null=None, linestyles = None, order=None, savepath=None, plotall=False):
    # todone Rico save routine

    fig, ax = plt.subplots(figsize=(6.4,2.4))

    datapoints = range(len(train))
    if null is not None:
        plt.semilogy(datapoints, train, color=colors[0], linestyle=linestyles[0], zorder=order[0], label='attempt train')
        plt.semilogy(datapoints, test, color=colors[1], linestyle=linestyles[1], zorder=order[1], label='attempt test')
        plt.semilogy(datapoints, null, color=colors[2], linestyle=linestyles[2], zorder=order[2], label='attempt nullset')
    else:
        plt.semilogy(datapoints, train, color=colors[0], linestyle=linestyles[0], zorder=order[0], label='attempt train')
        plt.semilogy(datapoints, test, color=colors[1], linestyle=linestyles[1], zorder=order[1], label='attempt test')

    count = 0
    for best in bestcount:
        if count == 0:
            plt.semilogy(datapoints[best], train[best], color='m', marker='o', zorder=order[0], linestyle='None', label='chosen train')
            plt.semilogy(datapoints[best], test[best], color='c', marker='o', zorder=order[1], linestyle='None', label='chosen test')
            count += 1
        else:
            plt.semilogy(datapoints[best], train[best], color='m', marker='o', linestyle='None', zorder=order[0])
            plt.semilogy(datapoints[best], test[best], color='c', marker='o', linestyle='None', zorder=order[0])


    plt.title(title)
    plt.xlabel(x, fontsize=9)
    plt.ylabel(y, fontsize=9)
    if ylim is not None:
        plt.ylim(ylim)
    tic = []
    for ii in range(0, len(datapoints) - 1, 10):
        tic.append(ii)
    plt.xticks(tic, visible=True, rotation="horizontal")
    # plt.yticks([my_yticks[1], my_yticks[-1]], visible=True, rotation="horizontal")
    plt.grid()
    plt.legend(fontsize=7)
    plt.autoscale(enable=True, axis='x', tight=True)



    if savepath is not None:
        plt.savefig(savepath, bbox_inches='tight', format='pdf')
        # plt.savefig(savepath, format='pdf')
    elif plotall:
        plt.show()





if __name__ == '__main__':
    num = 72
    dim = 0
    x = 'epochs'
    y = 'Loss'
    title = 'Dim {dim}, data point {ii}'
    dims = ['0-dim', '1-dim', '2-dim']
    colors = ['r', 'g', 'b']
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    linestyles = ['solid', 'dashed', (0, (3, 5, 1, 5))]
    layer_order = [10, 5, 0]
    # with open('../Learn_Master/checkpoints/cresults_s05ep_pen01.pkl', 'rb') as f1:
    #     graph_dict = pickle.load(f1)

    with open('../Learn_Master/predictions/chk/chkprediction_s10ep_optim01_2.pkl', 'rb') as f:
        pred = pickle.load(f)

    with open('../Learn_Master/predictions/chk/chkresults_s10ep_optim01_2.pkl', 'rb') as f:
        train_res = pickle.load(f)

    train = []
    test = []
    null = []
    ftr = []
    fte = []
    fnu = []

        # sett = graph_dict['improved_meta_set{}'.format(ii)]
    sett = train_res[0]
    tr = sett[0]
    te = sett[1]
    nu = sett[2]; nu = np.average(nu, axis=2)
        # ftr.append(tr[dim][-1])
        # fte.append(te[dim][-1])
        # fnu.append(nu[dim][-1])


    # def plot_metaloss(train, test, title, colors, x, y, null=None, linestyles=None, order=None, dim=0, merge=False):





    # for d in range(4):
    #     y = 'dim {} loss'.format(d)
    #     plot_metaloss(tr, te, '', color_list, x, y, null=nu, linestyles=linestyles, order=layer_order, dim=d, merge=False)



# plot_loss
#     train = []; test = []; null = []
    # ftr = []; fte = []; fnu = []
    # for ii in range(1, num+1):
    #     # sett = graph_dict['improved_meta_set{}'.format(ii)]
    #     sett = train_res
    #     tr = sett[0]; train.append(tr)
    #     te = sett[1]; test.append(te)
    #     nu = sett[2]; null.append(nu); nu = np.average(nu, axis=2)
    #     ftr.append(tr[dim][-1])
    #     fte.append(te[dim][-1])
    #     fnu.append(nu[dim][-1])
    #
    # # bestcount = graph_dict['improved_meta_set_bestcount']
    # # plot_metaloss(train, test, title, colors, x, y, null=null, linestyles=linestyles, order=layer_order, merge=True, dim=dim)
    # ftitle = 'Final Loss For Dimension {}'.format(dim)
    # x = 'Runs'
    # ylim = (5e-3, 5e-2)
    # plot_final_loss(ftr, fte, ftitle, x, y, colors, bestcount, ylim, null=None, linestyles=linestyles, order=layer_order)



# pred v actual
    with open('../Learn_Master/predictions/chk/chkprediction_s10ep_optim01_0.pkl', 'rb') as f:
        pred_act = pickle.load(f)

    p = pred_act[0]
    p = np.asarray(p)
    p = np.reshape(p, (477, 4))
    a = pred_act[1]


    # act = a[:, 0]
    # act = np.reshape(act, (act.shape[0], 1))

    act = a[0]
    nbars = 50
    for d in range(4):
        pred = p

        y = 'dimension {} component'.format(d)
        title = ''
        comparative_bar_plot(pred[:50, :], act[:50, :], nbars, y, title, dim=d, sort='')


    pred = pred[:, :3]
    act = act[:, :3]
    scatterplot3d(pred, act, type='vector', num_points=5, colors=color_list, styles=[linestyles[0], linestyles[1]])
# comparative_bar_plot(prediction, actual, nbars, y, title, dim=1, width=0.35, sort='ascending', plotall=False)
# scatterplot3d(prediction, actual, type='', num_points=20, colors=[], styles=[], seed=24)



# pre-meta-optimization script


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
    #
    # plot_one_loss(graph_dict, x, y, title, colors, dicts_wanted=dicts_wanted, linestyles=linestyles, order=layer_order, shift=shift)
    #
    #
    #
    # # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\Rico-Corpus\model_10000ep_2dims\BOWavg_rico\prediction.pkl', 'rb') as f3:
    # #     pred = pickle.load(f3)
    # #
    #
    #
    # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Ordered_Data\Rico-Corpus\model_10000ep_30dims\BOWsum\w100\train_labels.pkl', 'rb') as f4:
    #     act = pickle.load(f4)
    #
    #
    #
    #
    # # pred00_dict = graph_dict[f'rnn_100ep_pred_rico_BOWsum_w_100_3dim_100rnntanh-20tanh_00']
    # # pred01_dict = graph_dict[f'rnn_100ep_pred_rico_BOWsum_w_100_3dim_100rnntanh-20tanh_01']
    # # pred02_dict = graph_dict[f'rnn_100ep_pred_rico_BOWsum_w_100_3dim_100rnntanh-20tanh_02']
    # # #
    # # pred00 = pred00_dict['prediction']; pred00 = np.reshape(pred00, (pred00.shape[1], 1))
    # # pred01 = pred01_dict['prediction']; pred01 = np.reshape(pred01, (pred01.shape[1], 1))
    # # pred02 = pred02_dict['prediction']; pred02 = np.reshape(pred02, (pred02.shape[1], 1))
    # # #
    # # pred = np.concatenate((pred00, pred01, pred02), axis=1)
    #
    # pred_dict = graph_dict[f'rnn_500ep_pred_rico_BOWsum_w_100_3dims_rnntanh240-tanh111_5st']
    # pred = pred_dict['prediction']
    #
    # title = f'Prediction vs. Actual\n rnn_500ep_pred_rico_BOWsum_w_100_3dims_rnntanh240-tanh111_5st'
    #
    # num = 50
    # ordered_components(pred, act, num, colors)
    #
    # nbars = 100
    # ylab = f'Dimension 0{dim} Component'
    #
    # # comparative_bar_plot(pred, act, nbars, ylab, title, dim=dim, plotall=False)
    #
    # color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # scatterplot3d(pred, act, type='vector', colors=color_list, num_points=5, styles = ['-', 'dashed'], seed=3)

