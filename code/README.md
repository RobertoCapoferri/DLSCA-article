# Code

This is the code that was used to produce the results in `A comparison of deep learning approaches for power-based side-channel attacks against 32-bit micro-controller`.

- in the `modeling` folder there is the implementation of the neural network and the hyperparameter tuning algorithm.
- in the `scripts` folder there are the scripts to do the whole tuning, training and testing procedure, including the production of the graphs
- the `utils` folder contains support code for the rest of the project

## Usage

In the `constants.py` file there are two variables specifying the paths to use for all the other code, change these if needed:

- `PC_TRACES_PATH` specifies where to find the dataset, defaults to '$HOME/dataset'
- `RESULTS_PATH` is the path where all the models and graphs will be saved '$HOME/results'

All the procedure can be reproduced using the code in the `scripts` folder.

We ran our experiments in a virtual machine, with 20 CPU cores available, 128 GB of RAM and 2x Nvidia A100 40GB GPUS. Using multiple GPUs speeds up the work since more models can be tuned and trained in parallel.