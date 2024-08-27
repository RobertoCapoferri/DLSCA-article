# Basic
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import L2

# Custom
import sys
sys.path.insert(0, '../utils')
import constants


class Network():

    """
    Neural Network implementation.

    Attributes:
        - model_type (str):
            Type of Neural Network to implement.
        - hp (dict):
            Neural Network hyperparameters.
        - model (keras.models.Sequential):
            Neural Network model.
            By default, it is an empty Sequential object.
        - callbacks (keras.callbacks list):
            List of callbacks to use during model training.
            By default, EarlyStopping and ReduceLROnPlateau are considered.

    Methods:
        - add_checkpoint_callback:
            Adds ModelCheckpoint to the list of callbacks.
        - build model:
            Generates the Neural Network model.
        - _compute_key_preds:
            Converts target-predictions into key-predictions.
        - _compute_final_rankings:
            Generates the final ranking of all possible key-bytes.
        - ge:
            Computes the Guessing Entropy of an attack.
    """

    def __init__(self, model_type, hp):

        """
        Class constructor: takes as input all class attributes and generates a
        Network object.
        """

        self.model_type = model_type
        self.hp = hp
        self.model = Sequential()

        self.callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=1e-7),
        ]


    def add_checkpoint_callback(self, model_path):

        """
        Adds ModelCheckpoint to the list of callbacks.

        ModelCheckpoint allows to save the best-performing model during training
        (performance is given by the validation loss (the lower, the better)).

        Parameters:
            - model_path (str):
                Path to where to store the model (model is a H5 file).
        """

        self.callbacks.append(
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True
            )
        )


    def build_model(self):

        """
        Generates the Neural Network model adding layer to the default empty
        Sequential object.

        Different models can be generated:
        The name must contain all of the parts
        e.g. MLP_ptx_key builds an MLP which also takes the ptx in input
        and has key as target
        """

        if self.model_type: # Generate an MLP

            #### Architecture: ####
            # Input Dense         #
            # Input BatchNorm     #
            #                     #
            # repeat(             #
            #   Hidden Dropout    #
            #   Hidden Dense      #
            #   Hidden BatchNorm  #
            # )                   #
            #                     #
            # Output Dropout      #
            # Output Dense        #
            # Output BatchNorm    #
            #######################

            # Input Dense
            if 'ptx' in self.model_type:
                self.model.add(Dense(constants.TRACE_LEN + 1, activation='relu', name='Input'))
            else:
                self.model.add(Dense(constants.TRACE_LEN, activation='relu', name='Input'))

            # Input BatchNorm
            self.model.add(BatchNormalization(name='InputBatchNorm'))

            # Hidden
            for i in range(self.hp['hidden_layers']):
                # Hidden Dropout
                self.model.add(Dropout(self.hp['dropout_rate'], name=f'HiddenDropout{i}'))
                # Hidden Dense
                self.model.add(Dense(
                    self.hp['hidden_neurons'],
                    activation='relu',
                    kernel_regularizer=L2(self.hp['l2']),
                    name=f'HiddenDense{i}')
                )
                # Hidden BatchNorm
                self.model.add(BatchNormalization(name=f'HiddenBatchNorm{i}'))

            # Output
            # Output Dropout
            self.model.add(Dropout(self.hp['dropout_rate'], name='OutputDropout'))
            # Output Dense with BatchNorm before activation
            if 'HW_SO' in self.model_type:
                self.model.add(Dense(9, name='Output')) #change to 9 if use HW leakage model
            else:
                self.model.add(Dense(256, name='Output')) #change to 9 if use HW leakage model
            self.model.add(BatchNormalization(name='OutputBatchNorm'))
            self.model.add(Activation('softmax', name='Softmax'))

            # Compilation
            lr = self.hp['learning_rate']
            if self.hp['optimizer'] == 'adam':
                opt = Adam(learning_rate=lr)
            elif self.hp['optimizer'] == 'rmsprop':
                opt = RMSprop(learning_rate=lr)
            else:
                opt = SGD(learning_rate=lr)

            self.model.compile(
                optimizer=opt,
                loss='categorical_crossentropy',
                metrics=['accuracy'] # Automatically detected as "categorical_accuracy"
                                     # if "categorical_crossentropy" is used (one-hot encoding)
            )

        else:
            raise RuntimeError(f'"{self.model_type}" is not a valid model')
