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



def plot_pred_v_actual(prediction, actual):
    x1 = []
    y1 = []
    x2 = []
    y2 =[]
    shapes = ['.', 'o', 'v', '>', '1', '4', 's', '+', 'X', 'd']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'springgreen', 'tan', 'slategrey']
    for ii in range(10):
        num = np.random.randint(0, len(prediction))
        x1.append(prediction[num, 0])
        y1.append(prediction[num, 1])
        x2.append(actual[num, 0])
        y2.append(actual[num, 1])

        plt.plot(prediction[num, 0], prediction[num, 1], marker=shapes[ii], label=f'Prediction {ii}', color=colors[ii])
        plt.plot(actual[num, 0], actual[num, 1], marker=shapes[ii], label=f'Actual {ii}', color=colors[ii])

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
    plt.legend(loc='best')#, bbox_to_anchor=(0.23, 0.62, 0.5, 0.5))
    plt.show()


if __name__ == '__main__':
    x = 'Epochs'
    y = 'Loss'
    title = 'FF BOW-avg Shuffled Sciart'
    colors = ['r', 'g', 'b']
    linestyles = ['solid', 'dashed', 'dotted']
    layer_order = [10, 5, 0]
    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Metastimuli-Learn\Atom-FFNN\graphing_data.pkl', 'rb') as f1:
        prob_graph_dict = pickle.load(f1)

    with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Metastimuli-Learn\Atom-FFNN\regress_graph_data.pkl', 'rb') as f2:
        regress_graph_dict = pickle.load(f2)

    # plot_loss(regress_graph_dict, 'Epochs', 'Loss', 'Plot1')

    # with open(r'C:\Users\liqui\PycharmProjects\Generation_of_Novel_Metastimulus\Lib\Misc_Data\prediction.pkl', 'rb') as f3:
    #     pred = pickle.load(f3)
    #
    # with open(r'C:\Users\liqui\PycharmProjects\Word_Embeddings\Lib\Data\train_labels_proj.pkl', 'rb') as f4:
    #     act = pickle.load(f4)
    #
    # plot_pred_v_actual(pred, act)


    dicts_wanted = [
        'ff_100ep_train_sciart_BOWavg_shuff_proj',
        'ff_100ep_test_sciart_BOWavg_shuff_proj',
        'ff_100ep_nullset_sciart_BOWavg_shuff_proj'
                    ]
    shift = [
        'ff_100ep_test_sciart_BOWavg_shuff_proj',
        'ff_100ep_nullset_sciart_BOWavg_shuff_proj'
    ]
    plot_one_loss(regress_graph_dict, x, y, title, colors, dicts_wanted=dicts_wanted, linestyles=linestyles, order=layer_order, shift=shift)


