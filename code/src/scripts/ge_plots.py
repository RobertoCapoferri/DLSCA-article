"""compute GE against different devices on all the models found in folder
usage: python3 ge_plots.py /path/to/folder
attacks against different devices using fixed key dataset
tries to guess target from filename
"""

# Basics
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# increase font size
plt.rcParams.update({'font.size': 27})

# Custom
import sys
sys.path.insert(0, '../utils')
import results
from utils import ensure_path
import constants
from data_loader import DataLoader
sys.path.insert(0, '../modeling')

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs

# Training Parameters
BYTE_IDX = 5    # attack 6th byte
VERBOSE = 1     # model.fit verbosity level 0, 1, 2

# Paths
BASE = sys.argv[1]
RES_ROOT = os.path.join(BASE, 'ge_plots_comparison')
assert ensure_path(BASE), 'path doesn\'t exist'
assert ensure_path(RES_ROOT), 'path doesn\'t exist'
print(f'paths: \n{BASE}\n{RES_ROOT}')

# utility functions
def plot_ge(ge, output_path, show=False):

    """
    Plots the provided GE vector.

    Parameters:
        - ge (np.array):
            GE vector to plot.
        - output_path (str):
            Absolute path to the PNG file containing the plot.
    """

    # Plot GE
    f, ax = plt.subplots(figsize=(15,8))

    v_value = results.min_att_tr(ge)
    if v_value > len(ge):
        ax.axvline(v_value + 300, color='r', linestyle='--', linewidth=3, label=f'GE = {len(ge)}+')
    else:
        ax.axvline(v_value, color='r', linestyle='--', linewidth=3, label=f'GE = {v_value}')
    ax.plot(range(1, len(ge)+1), ge, marker='o', color='b')

    x_label_step = 30
    # set limits in the representation
    ax.set_xlim([0,len(ge)])


    ax.set_title(f'Guessing Entropy')
    ax.set_xticks(range(0, len(ge)+1, x_label_step), labels=range(0, len(ge)+1, x_label_step))
    ax.set_xlabel('Number of traces')
    ax.set_ylabel('GE')
    ax.set_yticks(range(0, 100+1, 10), labels=range(0, 100+1, 10))
    ax.grid(alpha=0.2)


    ax.legend(loc='best')
    f.savefig(
        output_path,
        bbox_inches='tight',
        dpi=600
    )

    if show:
        plt.show()

    plt.close(f)

def plot_comparison(ge_results: dict[str, np.ndarray], output_path: str, show=False):

    """
    Plots the comparison of all ge results.

    Parameters:
        - ge (np.array):
            GE vector to plot.
        - output_path (str):
            Absolute path to the PNG file containing the plot.
    """

    # Plot GE
    f, ax = plt.subplots(figsize=(15,8))

    colors = {
        'diff_devs_diff_key': 'r',
        'diff_devs_key_1': 'r',
        'same_devs_diff_key': 'g',
        'same_devs_key_1': 'g',
        'diff_devs_same_key': 'b',
        'diff_devs_key_0': 'b',
    }
    for label, ge in ge_results.items():

        v_value = results.min_att_tr(ge)
        if v_value <= len(ge):
            ax.axvline(v_value, color=colors[label], linestyle='--', linewidth=3)
            tag=f'{label} GE = {v_value}'
        else:
            tag=f'{label} GE = {len(ge)}' + '+'
        # ax.plot(ge, marker='o', color=colors[label])
        ax.plot(range(1, len(ge)+1), ge, marker='o', color=colors[label], label=tag)

        x_label_step = 30
        # set limits in the representation
        ax.set_xlim([0,len(ge)])

        ax.set_title(f'Guessing Entropy')
        ax.set_xticks(range(0, len(ge)+1, x_label_step), labels=range(0, len(ge)+1, x_label_step))
        ax.set_xlabel('Number of traces')
        ax.set_ylabel('GE')
        ax.set_yticks(range(0, 100+1, 10), labels=range(0, 100+1, 10))
        ax.grid(alpha=0.2)

    ax.legend(loc='best')
    f.savefig(
        output_path,
        bbox_inches='tight',
        dpi=600
    )
    f.savefig(
        f'{output_path}.svg',
        bbox_inches='tight',
        dpi=600
    )

    if show:
        plt.show()

    plt.close(f)

b = 5  # byte index to attack

for file in os.listdir(BASE):
    same_devs = []
    diff_devs = ['D1', 'D2', 'D3']
    target = ''
    print(file)
    if file.endswith('.h5'):
        MODEL_FILENAME = os.path.join(BASE, file)
        print(f'found model {file}')
        if 'D1' in file:
            diff_devs.remove('D1')
            same_devs.append('D1')
        if 'D2' in file:
            same_devs.append('D2')
            diff_devs.remove('D2')
        if 'D3' in file:
            same_devs.append('D3')
            diff_devs.remove('D3')
        if 'SBOX_OUT' in file:
            target = 'SBOX_OUT'
        if 'KEY' in file:
            target = 'KEY'
        if 'HW_SO' in file:
            target = 'HW_SO'

        assert len(diff_devs) > 0, 'cannot figure out device to test'
        assert target != '', 'cannot figure out target'

        #### DATASETS FOR TEST
        test_files = {}
        test_files['diff_devs_diff_key'] = [f'{constants.PC_TRACES_PATH}/{dev}_fixed_key_test_resampled.trs'
                for dev in diff_devs]
        test_files['same_devs_diff_key'] = [f'{constants.PC_TRACES_PATH}/{dev}_fixed_key_test_resampled.trs'
                for dev in same_devs]
        test_files['diff_devs_same_key'] = [f'{constants.PC_TRACES_PATH}/{dev}_fixed_key_resampled.trs'
                for dev in diff_devs]


        # save GE for plotting comparison later
        ge_results = {}
        plot = True
        for case, files in test_files.items():
            # relabel only if i test against random
            if ('random' in MODEL_FILENAME and 'transfer' not in MODEL_FILENAME) or \
                ('fixed' in MODEL_FILENAME and 'transfer' in MODEL_FILENAME):
                case = case.replace('diff_key', 'key_1').replace('same_key', 'key_0')
            # log
            GE_FILENAME = os.path.join(RES_ROOT, case + file.replace('model', 'ge').replace('.h5', ''))
            if os.path.isfile(GE_FILENAME + '.png'):
                print(f'skipping {file} because it already has a plot in {GE_FILENAME}')
                plot = False
                break
            print(f'found model {file}, saving results in {GE_FILENAME}')
            # print(f'testing devices = {diff_devs}')

            # do the testing
            test_dl = DataLoader(
                files,
                tot_traces=30000,
                target=target,
                byte_idx=b
            )
            x_test, _, pbs_test, tkb_test = test_dl.load()


            # same scaling as above
            # tkb is just one for fixed
            print(len(tkb_test), len(tkb_test), tkb_test, tkb_test[0], sep='\n')
            scaler = MinMaxScaler()
            scaler.fit(x_test)
            x_test = scaler.transform(x_test)
            if 'no_vs_ptx' in file or ('ptx' in file and 'ptx_vs_no' not in file):
                if 'scaled' in file:
                    pbs_test_in = pbs_test / 255
                    x_test = np.append(x_test, pbs_test_in, axis=1)
                else:
                    x_test = np.append(x_test, pbs_test, axis=1)

            # Compute GE
            test_model = load_model(MODEL_FILENAME)

            ##### FIXED
            ge = results.ge(
                model=test_model,
                x_test=x_test,
                ptx_bytes=pbs_test,
                true_key_byte=tkb_test[0],
                n_exp=100,
                target=target
            )
            plot_ge(ge[:300], GE_FILENAME)

            # save data to csv
            np.savetxt(GE_FILENAME + '.csv', ge, delimiter=',')

            # append to comparison graph()
            ge_results[case] = ge[:300]

        if plot:
            GE_FILENAME = os.path.join(RES_ROOT, 'comparison' + file.replace('model', 'ge').replace('.h5', ''))
            plot_comparison(ge_results, GE_FILENAME)