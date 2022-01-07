from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.express as px
from scipy import stats as scst

from ptools.lipytools.stats import stats_pd, msmx
from ptools.neuralmess.get_tf import tf


# tensorboard plotter
class TBPoltter:

    def __init__(
            self,
            folder: str):

        self.__writer = tf.summary.FileWriter(
            logdir=         folder,
            flush_secs=     10)

    def log(
            self,
            value,
            tag: str,
            step: int):

        vsumm = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.__writer.add_summary(vsumm, step)


def histogram(
        val_list :list or np.array,
        name=               'values',
        rem_nstd :float=    0.0,    # removes values out of N*stddev
        msmx_stats=         True,   # prints minimal stats
        pandas_stats=       False,  # prints pandas extended stats
        density=            True,
        bins: int or None=  None,   # automatic for None
        save_FD :str=       None):

    if not val_list: print('WARNING: Can not prepare histogram for empty val_list!')
    else:
        if msmx_stats: print(f' > "{name}": {msmx(val_list)["string"]}')
        if pandas_stats: print(f' > stats with pandas for "{name}":\n{stats_pd(val_list)}')

        if rem_nstd:
            stats = msmx(val_list)
            std = stats['std']
            mean = stats['mean']
            val_list = [val for val in val_list if mean - rem_nstd * std < val < mean + rem_nstd * std]

            if pandas_stats:
                print(f'\n > after removing {rem_nstd} stddev:')
                print(stats_pd(val_list))

        if not bins:
            bins = len(set(val_list))
            if bins>50: bins = 50

        plt.clf()
        n, x, _ = plt.hist(val_list, label=name, density=density, bins=bins, alpha=0.5)

        # try build density, for some val_list it is not possible
        try:
            density = scst.gaussian_kde(val_list)
            plt.plot(x, density(x))
        except: pass

        plt.legend(loc='upper right')
        plt.grid(True)
        if save_FD:
            if not os.path.isdir(save_FD): os.makedirs(save_FD, exist_ok=True)
            plt.savefig(f'{save_FD}/{name}.png')
        else:
            plt.show()


def two_dim(
        y :list or np.array,            # two(yx) or one(y) dim list or np.array
        x :list or np.array=    None,
        name=                   'values',
        save_FD: str =          None,
        xlogscale=              False,
        ylogscale=              False,
        legend_loc=             'upper right'):

    if type(y) is list: y = np.array(y)
    if x is None:
        if len(y.shape) < 2: x = np.arange(len(y))
        else:
            x = y[:, 1]
            y = y[:, 0]

    plt.clf()
    plt.plot(x, y, label=name)
    if xlogscale: plt.xscale('log')
    if ylogscale: plt.yscale('log')
    plt.legend(loc=legend_loc)
    plt.grid(True)
    if save_FD: plt.savefig(f'{save_FD}/{name}.png')
    else:       plt.show()


def three_dim(
    xyz: list, # list of (x,y,z) or (x,y,z,val)
    name=               'values',
    x_name=             'x',
    y_name=             'y',
    z_name=             'z',
    val_name=           'val',
    save_FD: str =      None):

    # expand to 3 axes + val (3rd axis data)
    if len(xyz[0])<4:
        new_xyz = []
        for e in xyz: new_xyz.append(list(e) + [e[-1]])
        xyz = new_xyz

    df = pd.DataFrame(
        data=       xyz,
        columns=    [x_name,y_name,z_name,val_name])

    std = df[val_name].std()
    mean = df[val_name].mean()
    off = 2*std
    cr_min = mean - off
    cr_max = mean + off

    fig = px.scatter_3d(
        data_frame=     df,
        title=          name,
        x=              x_name,
        y=              y_name,
        z=              z_name,
        color=          val_name,
        range_color=    [cr_min,cr_max],
        opacity=        0.7,
        width=          700,
        height=         700)

    if save_FD:
        file = f'{save_FD}/{name}_3Dplot.html'
        fig.write_html(file, auto_open=False if os.path.isfile(file) else True)
    else: fig.show()



if __name__ == '__main__':

    # *********************** histogram example
    histogram(np.array([1, 4, 5, 5, 3, 3, 5, 65, 32, 45, 5, 5, 6, 33, 5]))


    # *********************** two_dim example
    n = 100
    x = (np.arange(n)-n//2)/(n/2/np.pi/3)
    y = np.sin(x)

    two_dim(y,x)
    two_dim(list(zip(y,x)))
    two_dim(y)


    # *********************** three_dim example
    from ptools.neuralmess.layers import positional_encoding
    width = 1#5

    pe = positional_encoding(90, width, 0.9, 7, verb=1)
    pe = np.squeeze(pe)

    print(pe.shape)
    if width == 1:
        two_dim(pe)
    else:
        two_dim(pe[:, 0]) # first
        #two_dim(pe[:, 1])
        #two_dim(pe[:,-2])
        two_dim(pe[:,-1]) # last

    if width == 1:
        xyz = []
        for rix in range(pe.shape[0]):
            xyz.append([rix, 0, pe[rix]])
    else:
        xyz = []
        for rix in range(pe.shape[0]):
            for eix in range(pe.shape[1]):
                xyz.append([rix,eix,pe[rix,eix]])

    three_dim(xyz)