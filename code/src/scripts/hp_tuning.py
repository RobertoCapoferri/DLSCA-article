# Basics
import json
import time
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Custom
import sys
sys.path.insert(0, '../utils')
import helpers
import constants
import visualization as vis
from utils import ensure_path
from data_loader import SplitDataLoader
sys.path.insert(0, '../modeling')
from hp_tuner import HPTuner

# Suppress TensorFlow messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # 1 for INFO, 2 for INFO & WARNINGs, 3 for INFO & WARNINGs & ERRORs
# prevent oom error
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # allocate memory over time, so i can see the usage in real time

N_MODELS = 15
N_GEN = 20
EPOCHS = 100
HP = {
    'hidden_layers':  [1, 2, 3, 4, 5],
    'hidden_neurons': [100, 200, 300, 400, 500],
    'dropout_rate':   [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'l2':             [0.0, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    'optimizer':      ['adam', 'rmsprop', 'sgd'],
    'learning_rate':  [5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
    'batch_size':     [128, 256, 512, 1024]
}

def main():

    """
    Performs hyperparameter tuning for an MLP model with the specified settings.
    Settings parameters (provided in order via command line):
        - use_ptx: 0 or 1, whether to use ptx in training or not
        - random: string random or fixed, whether to use random key dataset or not
        - target: string, SBOX_OUT or KEY or HW_SO
        - train_devs: Devices to use during training

    The result is a JSON file containing the best hyperparameters.
    """

    use_ptx = sys.argv[1]
    dataset = sys.argv[2]
    TARGET = sys.argv[3]
    train_devs = sys.argv[4:]
    b = 5
    use_ptx = True if int(use_ptx) == 1 else False
    assert dataset == 'random' or dataset == 'fixed'
    TARGET = str(TARGET)
    assert TARGET == 'KEY' or TARGET == 'SBOX_OUT' or TARGET == 'HW_SO'

    # tuning done with 50k traces for speed
    tot_traces = 50000

    RES_ROOT = f'{constants.RESULTS_PATH}/{TARGET}/{len(train_devs)}d/{dataset}_key'
    IMAGES = RES_ROOT + '/plots'
    # make dir if non existant
    assert ensure_path(RES_ROOT), 'res_root broken'
    assert ensure_path(IMAGES), 'images broken'

    id_train = ''.join(train_devs) + f'{TARGET}'
    if use_ptx:
        id_train += '_ptx'
    train_files = [f'{constants.PC_TRACES_PATH}/{dev}_{dataset}_key_resampled.trs'
                    for dev in train_devs]

    LOSS_HIST_FILE = RES_ROOT + f'/{id_train}_hp_loss_hist_data.csv'
    ACC_HIST_FILE = RES_ROOT + f'/{id_train}_hp_acc_hist_data.csv'
    HISTORY_PLOT = IMAGES + f'/{id_train}_hp_tuning_history.svg'
    HP_PATH = f'{RES_ROOT}/{id_train}_hp.json'


    # Get data
    train_dl = SplitDataLoader(
        train_files,
        tot_traces=tot_traces,
        train_size=0.9,
        target=TARGET,
        byte_idx=b
    )
    train_data, val_data = train_dl.load()
    x_train, y_train, ptx_train, _ = train_data
    x_val, y_val, ptx_val, _ = val_data

    # scale data in [0-1] range
    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    model_type = f'MLP_{TARGET}'
    if use_ptx:
        #add ptx bytes to train and validation sets
        # ptx_train = ptx_train / 255
        x_train = np.append(x_train, ptx_train, axis = 1)
        # ptx_val = ptx_val / 255
        x_val = np.append(x_val, ptx_val, axis = 1)
        model_type += '_ptx'

    print(f'model: {model_type} (use ptx = {use_ptx})')

    # HP Tuning via Genetic Algorithm
    hp_tuner = HPTuner(
        model_type=model_type,
        hp_space=HP,
        n_models=N_MODELS,
        n_epochs=EPOCHS
    )

    best_hp = hp_tuner.genetic_algorithm(
        n_gen=N_GEN,
        selection_perc=0.3,
        second_chance_prob=0.2,
        mutation_prob=0.2,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val
    )
    print(best_hp)

    # Save history data to .CSV files
    b_history = hp_tuner.best_history
    actual_epochs = len(b_history['loss']) # Early Stopping can make the actual
                                            # number of epochs different from the original one

    # Loss
    loss_data = np.vstack(
        (
            np.arange(actual_epochs)+1, # X-axis values
            b_history['loss'], # Y-axis values for 'loss'
            b_history['val_loss'] # Y-axis values for 'val_loss'
        )
    ).T
    helpers.save_csv(
        data=loss_data,
        columns=['Epochs', 'Loss', 'Val_Loss'],
        output_path=LOSS_HIST_FILE
    )

    # Accuracy
    acc_data = np.vstack(
        (
            np.arange(actual_epochs)+1, # X-axis values
            b_history['accuracy'], # Y-axis values for 'loss'
            b_history['val_accuracy'] # Y-axis values for 'val_loss'
        )
    ).T
    helpers.save_csv(
        data=acc_data,
        columns=['Epochs', 'Acc', 'Val_Acc'],
        output_path=ACC_HIST_FILE
    )

    # Plot training history
    vis.plot_history(b_history, HISTORY_PLOT)

    with open(HP_PATH, 'w') as jfile:
        json.dump(best_hp, jfile)


if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Elapsed time: {((time.time() - start) / 3600):.2f} h')
