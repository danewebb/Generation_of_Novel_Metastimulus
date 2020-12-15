from matplotlib import pyplot as plt
import pickle
import numpy as np



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



def plot_pred_v_actual(prediction, actual, subx, suby, linestyles=[], order=[]):
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
    plt.show()
    for jj in range(subx):
        for kk in range(suby):
            ax[jj, kk].quiver((0, 0), (0, 0), ac[0, mm], ac[1, mm], scale=.1, color=colors[0])
            ax[jj, kk].quiver((0, 0), (0, 0), pr[0, mm], pr[1, mm], scale=.1, color=colors[1])
            # ax.set_xlim(-0.5, 0.5)
            # ax.set_ylim(-0.5, 0.5)

            # ax[jj, kk].quiver(*tail, (x1[mm], y1[mm]), color=colors[0], label='Prediction', linestyle=linestyles[0], zorder=order[0])
            # ax[jj, kk].quiver(*tail, (x2[mm], y2[mm]), color=colors[1], label='Actual', linestyle=linestyles[0], zorder=order[1])
            mm += 1


    # plt.legend()
    plt.show()


def plot_one_loss(graph_dict, x, y, title, colors, linestyles = [], dicts_wanted=[], order=[], shift=[]):
    col_idx = 0
    plt.figure()
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.semilogy()

    for key, value in graph_dict.items():
        if key in dicts_wanted:
            d = graph_dict[key]
            loss = d['loss']
            eps = range(d['epochs'])
            if key in shift:
                eps = eps[1:]
                loss = loss[:-1]
            plt.plot(eps, loss, label=key, color=colors[col_idx], linestyle=linestyles[col_idx], zorder=order[col_idx])
            col_idx += 1

    plt.grid()
    plt.legend(loc='best', bbox_to_anchor=(0.23, 0.62, 0.5, 0.5))
    plt.show()


if __name__ == '__main__':
    # x = 'Epochs'
    # y = 'Loss'
    # title = 'FF BOW-avg Shuffled Rico-corpus'
    # colors = ['r', 'g', 'b']
    # linestyles = ['solid', 'dashed', 'dotted']
    # layer_order = [10, 5, 0]
    # # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Metastimuli-Learn\Atom-FFNN\graphing_data.pkl', 'rb') as f1:
    # #     prob_graph_dict = pickle.load(f1)
    # #
    # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Metastimuli-Learn\Atom-FFNN\regress_graph_data.pkl', 'rb') as f2:
    #     regress_graph_dict = pickle.load(f2)
    #
    # # plot_loss(regress_graph_dict, 'Epochs', 'Loss', 'Plot1')
    #
    #
    # dicts_wanted = [
    #     'ff_50ep_train_rico_BOWavg_shuff_proj',
    #     'ff_50ep_test_rico_BOWavg_shuff_proj',
    #     'ff_50ep_nullset_rico_BOWavg_shuff_proj'
    #                 ]
    # shift = [
    #     'ff_50ep_test_rico_BOWavg_shuff_proj',
    #     'ff_50ep_nullset_rico_BOWavg_shuff_proj'
    # ]
    # plot_one_loss(regress_graph_dict, x, y, title, colors, dicts_wanted=dicts_wanted, linestyles=linestyles, order=layer_order, shift=shift)



    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\BOWsum_rico\prediction.pkl', 'rb') as f3:
        pred = pickle.load(f3)

    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Shuffled_Data_1\BOWsum_rico\train_labels.pkl', 'rb') as f4:
        act = pickle.load(f4)

    subx = 3
    suby = 3
    line = ['solid', 'dashed']
    ord = [5, 0]
    plot_pred_v_actual(pred, act, subx, suby, linestyles=line, order=ord)
