# Scripts

These are the script used to do tuning, training and testing.
The python scripts take arguments to specify which model to build and test. For instance if you want to tune a model with fixed key on the SBOX_OUT target without plaintext on one device the command would be

```
python3 ./hp_tuning.py 0 fixed SBOX_OUT D1
```

There are a number of helper scripts that will run all the configurations. They were made for our dual GPU setup but can be easily adapted. The names are self explainatory, plus there is the `run_all.sh` script that will run the whole procedure. Be careful since the tuning will take several days.