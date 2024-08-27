# DLSCA results on Riscure Pinata (ARM Cortex-M4F)

All the code is available in the `code` folder.

The results are divided by target (SBOX_OUT or HW_SO), number of devices (1d or 2d) and type of dataset (fixed or random key)
- `plots` contains the training/validation accuracy and loss
- `ge_plots_comparison` contains the plots of the evolution of Guessing Entropy (GE) for increasing number of traces
- `multi_ge`, if present, contains numerous runs of the GE computation to see the run to run variability

filenames are structured as <TRAINDEV><TARGET>_<SUFFIX> where
- `TRAINDEV` is the training device(s) used, can be `D1` or `D1D2`
- `TARGET` is either SBOX_OUT or HW_SO
- `SUFFIX` depends on the specific file
   - `hp.json` contains the hyperparamters for the specific model
   - `model` contains the model in hd5 format
   - if it contains `ptx` it means that it is referred to a network trained with trace data + plaintext information. The plaintext can be appended as is or scaled in [0,1]
