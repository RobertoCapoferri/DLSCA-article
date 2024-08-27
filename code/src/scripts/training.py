# Basics
import time
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import json
import numpy as np

# Custom
import sys
sys.path.insert(0, '../utils')
import constants
from utils import ensure_path
from data_loader import SplitDataLoader
sys.path.insert(0, '../modeling')
from network import Network

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs

# Training Parameters
EPOCHS = 200    # train over 100 epochs with early stopping
BYTE_IDX = 5    # attack 6th byte
VERBOSE = 1     # model.fit verbosity level 0, 1, 2

# utility functions
def plot_history(history, metric, output_path, show=False):

    """
    Plots the training history (train_loss vs val_loss, train_acc vs val_acc).

    Parameters:
        - history (dict):
            Train history.
        - metric (str):
            Metric to plot.
        - output_path (str):
            Absolute path to the .SVG file containing the plot.
    """

    f = plt.figure(figsize=(10,10))

    train_label = f'train_{metric}'
    val_label = f'val_{metric}'
    title = f'Train and Val {metric.title()}' # .title() upper-cases only the first letter

    plt.plot(history[metric], label=train_label)
    plt.plot(history[val_label], label=val_label)
    if metric == 'accuracy':
        plt.axhline(y=1/256, color='r', linewidth=3, linestyle='--', label='Random-Guesser Accuracy')
    plt.title(title)
    plt.ylabel(metric.title())
    plt.xlabel('Epochs')
    plt.legend()
    plt.grid()

    f.savefig(
        f'{output_path}.svg',
        bbox_inches='tight',
        dpi=600
    )
    f.savefig(
        f'{output_path}.png',
        bbox_inches='tight',
        dpi=600
    )
    if show:
        plt.show()

    plt.close(f)


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

    ax.plot(ge, marker='o', color='b')

    ax.set_title(f'Guessing Entropy')
    ax.set_xticks(range(0, len(ge)+1, 10), labels=range(0, len(ge)+1, 10))
    ax.set_xlabel('Number of traces')
    ax.set_ylabel('GE')
    ax.grid()

    f.savefig(
        f'{output_path}.svg',
        bbox_inches='tight',
        dpi=600
    )
    f.savefig(
        f'{output_path}.png',
        bbox_inches='tight',
        dpi=600
    )

    if show:
        plt.show()

    plt.close(f)

# train for each target
def main():
    """
    Settings parameters (provided in order via command line):
        - use_ptx: 0 or 1, whether to use ptx in training or not
        - random: string random or fixed, whether to use random key dataset or not
        - target: string, SBOX_OUT or KEY or HW_SO
        - train_devs: Devices to use during training
    """
    use_ptx = sys.argv[1]
    dataset = sys.argv[2]
    target = sys.argv[3]
    train_devs = sys.argv[4:]
    b = 5
    use_ptx = True if int(use_ptx) == 1 else False
    assert dataset == 'random' or dataset == 'fixed'
    target = str(target)
    assert target == 'KEY' or target == 'SBOX_OUT' or target == 'HW_SO'

    # tot_traces = 200000
    tot_traces = 15000

    RES_ROOT = f'{constants.RESULTS_PATH}/{target}/{len(train_devs)}d/{dataset}_key'
    IMAGES = RES_ROOT + '/plots'
    # Paths
    HP_PATH = RES_ROOT

    assert ensure_path(RES_ROOT), 'path doesn\'t exist'
    assert ensure_path(HP_PATH), 'path doesn\'t exist'
    assert ensure_path(IMAGES), 'path doesn\'t exist'

    id_train = ''.join(train_devs) + f'{target}'
    if use_ptx:
        id_train += '_ptx'
    train_files = [f'{constants.PC_TRACES_PATH}/{dev}_{dataset}_key_resampled.trs'
                    for dev in train_devs]
    # paths needed
    MODEL_FILENAME = os.path.join(RES_ROOT, f'{id_train}_model.h5')
    LOSS_HISTORY_FILENAME = os.path.join(IMAGES, f'{id_train}_loss_history')
    ACC_HISTORY_FILENAME = os.path.join(IMAGES, f'{id_train}_acc_history')

    # load hyperparamters
    hp_file = os.path.join(HP_PATH, id_train + '_hp.json')
    assert os.path.exists(hp_file), f'hp for {train_devs} (expected at {hp_file}) not found'
    with open(hp_file) as f:
        hp = json.load(f)
    # hp = {"hidden_layers": 4, "hidden_neurons": 300, "dropout_rate": 0.2, "l2": 0.001, "optimizer": "adam", "learning_rate": 0.005, "batch_size": 256}


    # log
    print(f'training devices = {train_devs}')

    # load training data
    train_dl = SplitDataLoader(
        train_files,
        tot_traces=tot_traces,
        train_size=0.9,
        target=target,
        byte_idx=b
    )
    train_data, val_data = train_dl.load()
    x_train, y_train, ptx_train, _ = train_data
    x_val, y_val, ptx_val, _ = val_data

    # scale data to [0-1] range
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    model_type = f'MLP_{target}'
    if use_ptx:
        #add ptx bytes to train and validation sets
        # ptx_train = ptx_train / 255
        x_train = np.append(x_train, ptx_train, axis = 1)
        # ptx_val = ptx_val / 255
        x_val = np.append(x_val, ptx_val, axis = 1)
        model_type += '_ptx'

    print(f'model: {model_type} (use ptx = {use_ptx})')

    # pass correct hyperparams to the network
    attack_net = Network(model_type, hp)
    attack_net.build_model()
    attack_net.add_checkpoint_callback(MODEL_FILENAME)

    print('Training start')
    # Training (with Validation)
    history = attack_net.model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=attack_net.hp['batch_size'],
        callbacks=attack_net.callbacks,
        verbose=VERBOSE
    ).history

    # save the history
    plot_history(history, 'loss', LOSS_HISTORY_FILENAME)
    plot_history(history, 'accuracy', ACC_HISTORY_FILENAME)

    print(attack_net.model.summary())

if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Elapsed time: {((time.time() - start) / 3600):.2f} h')