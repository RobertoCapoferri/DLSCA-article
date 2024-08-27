# Basics
import numpy as np

# Custom
from helpers import to_coords
import constants

# Typing
from typing import Callable


def _key(plaintext: np.ndarray, key: np.ndarray):
    return key

def _sbox_in(plaintext: np.ndarray, key: np.ndarray):
    # AddRoundKey
    sbox_in = plaintext ^ key
    return sbox_in

def _sbox_out(plaintext: np.ndarray, key: np.ndarray):
    # AddRoundKey
    sbox_in = plaintext ^ key
    # SubBytes
    rows, cols = to_coords(sbox_in)
    sbox_out = constants.SBOX_DEC[rows, cols]
    return sbox_out

def _hw_sbox_out(plaintext: np.ndarray, key: np.ndarray):
    # AddRoundKey
    sbox_in = plaintext ^ key
    # SubBytes
    rows, cols = to_coords(sbox_in)
    sbox_out = constants.SBOX_DEC[rows, cols]
    # HW Computation
    # hw = [bin(val).replace('0b', '').count('1') for val in sbox_out]
    # Python 3.10+
    hw = [val.bit_count() for val in sbox_out]
    return hw

def labels_from_key(plaintext: np.ndarray, key: np.ndarray, target: str):

    """
    Emulates AES-128 in order to generate target labels relative to the given
    plaintext and key.

    Parameters:
        - plaintext (np.array):
            Integer-version of the plaintext used during the encryption.
        - key (np.array):
            Integer-version of the key used during the encryption.
        - target (str):
            Target of the attack.

    Returns:
        - labels (np.array):
            Integer-version of the target labels.
    """

    actions: dict[str, Callable[[np.ndarray, np.ndarray], ]] = {
        'KEY': _key,
        'SBOX_IN': _sbox_in,
        'SBOX_OUT':_sbox_out,
        'HW_SO': _hw_sbox_out
    }
    generate_labels = actions[target] # action.get(target, _default_fun) to specify a default behavior in case of key not found

    labels = generate_labels(plaintext, key)

    return labels


def key_from_labels(ptx_byte, target: str) -> np.ndarray:

    """
    Recovers the key relative to each possible value of the attack target,
    given a plaintext byte.

    Parameters:
        - pltxt_byte (int):
            Single plaintext byte used during the encryption.
        - target (str):
            Target of the attack.

    Returns:
        - key_bytes (np.array):
            Key-bytes relative to each possible value of the attack target
    """

    possible_values = range(256)

    if target == 'SBOX_IN': # Directly sbox-in values
        sbox_in = np.array(possible_values)
    elif target == 'SBOX_OUT': # Sbox-out values: inverse-SubBytes needed
        rows, cols = to_coords(possible_values)
        sbox_in = constants.INV_SBOX_DEC[rows, cols]
    else:
        pass # Future implementations (HW, HD)

    # Inverse-AddRoundKey
    key_bytes = np.array(sbox_in ^ ptx_byte) # Shape: (1 x 256)

    return key_bytes
