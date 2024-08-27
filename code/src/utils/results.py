# Basics
import numpy as np
# Custom
import aes
import constants, helpers


def compute_key_preds(preds, key_bytes):

    """
    Converts target-predictions into key-predictions (key-byte).
    Returns the predictions sorted by key value  (from 0 to 255)

    Parameters:
        - preds (np.ndarray):
            Target-predictions (probabilities for each possible target value).
        - key_bytes (np.array):
            Key-bytes relative to each possible value in the target-predictions.
            Target-predictions cover all possible values (1 to 255), and
            each value leads to a differnt key-byte.

    Returns:
        - key_preds (np.array):
           Key-predictions (probabilities for each possible key-byte value).
           Sorted from key-byte=0 to key-byte=255.
    """

    # Associate each prediction with its relative key-byte
    association = list(zip(key_bytes, preds))

    # Sort the association w.r.t. key-bytes (0 to 255, for alignment within all traces)
    association.sort(key=lambda x: x[0])

    # Consider the sorted sbox-out predictons as key-byte predictons
    key_preds = list(zip(*association))[1]

    return key_preds


def compute_final_rankings(preds: np.ndarray, ptx_bytes: np.ndarray, target: str):

    """
    Generates the ranking of the key-bytes starting from key-predictions.

    Parameters:
        - preds (np.ndarray):
            Predictions relative to the target.
        - ptx_bytes (np.array):
            True plaintext bytes.
        - target (str):
            Target of the attack.

    Returns:
        - final_rankings (list):
            Ranking of the possible key-bytes (from the most probable to the
            least probable) for increasing number of traces.
    """

    if target == 'KEY':
        # If the target is 'KEY', then key_preds is directly preds (sampled_preds)
        # because it contains predictions related to each key-byte,
        # already in order (0 to 255)
        key_preds = np.array(preds) # preds is a tuple due to previous unzip
    elif target == 'HW_SO':
        # for each key hp, build the intermediate value
        intermediates = np.array([[constants.SBOX_DEC[helpers.to_coords(np.full((256), ps) ^ k)][0].bit_count()
                          for k in range(256)]
                          for ps in ptx_bytes])
        key_preds = np.array([ps[inter] for inter, ps in zip(intermediates, preds)])

    else:
        # SBOX-IN, SBOX-OUT need further computations
        # associate key byte to each sbox out
        key_bytes = [aes.key_from_labels(pb, target) for pb in ptx_bytes] # n_traces x 256

        # predictions sorted by the associated key value
        key_preds = np.array([compute_key_preds(ps, kbs)
                              for ps, kbs in zip(preds, key_bytes)]) # n_traces x 256

    # considering attacking a different trace as in independent experiment
    # then the combined probabilty is given by the product
    # to avoid multiplying by 0 use log and then add log(a) + log(b) = log(a*b)
    log_probs = np.log10(key_preds + 1e-22) # n_traces x 256

    cum_tot_probs = np.cumsum(log_probs, axis=0) # n_traces x 256

    indexed_cum_tot_probs = [list(zip(range(256), tot_probs))
                             for tot_probs in cum_tot_probs] # n_traces x 256 x 2

    sorted_cum_tot_probs = [sorted(el, key=lambda x: x[1], reverse=True)
                            for el in indexed_cum_tot_probs] # n_traces x 256 x 2

    # Generate the key-byte ranking for each element of the cumulative sum of
    # total probabilities
    final_rankings = [[el[0] for el in tot_probs]
                      for tot_probs in sorted_cum_tot_probs] # n_traces x 256

    # returns a list of lists, each list is the ranking of the key probabililties
    # each list positions tells how many attack traces where used
    # e.g. final_rankings[0] is the probabilities after 1 attack trace and is a list
    # of 256 values sorted by probability from highest to lowest of the key
    # e.g [123, 17, 32, ...] --> 123 most probable key byte
    return final_rankings


def ge(model, x_test, ptx_bytes, true_key_byte, n_exp, target):

    """
    Computes the Guessing Entropy of an attack as the average rank of the
    correct key-byte among the predictions.

    Parameters:
        - model (tensorflow.keras.Model):
            Classifier.
        - x_test (np.ndarray):
            Test data used to perform the attack.
        - pxt_bytes (np.array):
            Plaintext used during the encryption (single byte).
        - true_key_byte (int):
            Actual key used during the encryption (single byte).
        - n_exp (int):
            Number of experiment to compute the average value of GE.
        - target (str):
            Target of the attack.

    Returns:
        - ge (np.array):
            Guessing Entropy of the attack.
    """
    print(f'TRUEKEYBYTE = {true_key_byte}')
    tr_per_exp = int(x_test.shape[0] / n_exp)

    ranks_per_exp = []

    for i in range(n_exp):

        start = i * tr_per_exp
        stop = start + tr_per_exp

        # Consider a batch of test-data
        x_batch = x_test[start:stop]
        pltxt_bytes_batch = ptx_bytes[start:stop]

        # During each experiment:
        #   * Predict the target w.r.t. the current test-batch
        #   * Retrieve the corresponding key-predictions
        #   * Compute the final key-predictions
        #   * Rank the final key-predictions
        #   * Retrieve the rank of the correct key (key-byte)
        #
        # The whole process considers incrementing number of traces

        # Predict the target
        curr_preds = model.predict(x_batch)

        # Compute the final rankings (for increasing number of traces)
        final_rankings = compute_final_rankings(curr_preds, pltxt_bytes_batch, target)

        # Retrieve the rank of the true key-byte (for increasing number of traces)
        true_kb_ranks = np.array([kbs.index(true_key_byte)
                                  for kbs in final_rankings]) # 1 x n_traces

        ranks_per_exp.append(true_kb_ranks)

    # After the experiments, average the ranks
    ranks_per_exp = np.vstack(ranks_per_exp) # n_exp x n_traces
    ge = np.mean(ranks_per_exp, axis=0) # 1 x n_traces

    return ge


def ge_multi_keys(model, x_test, pltxt_bytes, true_key_bytes, n_exp, target):

    """
    Computes the Guessing Entropy of an attack as the average rank of the
    correct key-bytes among the predictions.

    Parameters:
        - model (tensorflow.keras.Model):
            Classifier.
        - x_test (np.ndarray):
            Test data used to perform the attack.
        - pltxt_bytes (np.array):
            Plaintext used during the encryption (single byte).
        - true_key_bytes (int):
            Actual keys used during the encryption (single byte).
        - n_exp (int):
            Number of experiment to compute the average value of GE.
        - target (str):
            Target of the attack.

    Returns:
        - ge (np.array):
            Guessing Entropy of the attack.
    """

    tr_per_exp = int(x_test.shape[0] / n_exp)

    ranks_per_exp = []

    for i in range(n_exp):

        print('Experiment number: ', i + 1)
        start = i * tr_per_exp
        stop = start + tr_per_exp

        # Consider a batch of test-data
        x_batch = x_test[start:stop]
        pltxt_bytes_batch = pltxt_bytes[start:stop]
        true_key_bytes_batch = true_key_bytes[start:stop]

        # During each experiment:
        #   * Predict the target w.r.t. the current test-batch
        #   * Retrieve the corresponding key-predictions
        #   * Compute the final key-predictions
        #   * Rank the final key-predictions
        #   * Retrieve the rank of the correct key (key-byte)
        #
        # The whole process considers incrementing number of traces

        # Predict the target
        curr_preds = model.predict(x_batch)

        # Compute the final rankings (for increasing number of traces)
        final_rankings = compute_final_rankings(curr_preds, pltxt_bytes_batch, target)

        # Retrieve the rank of the true key-byte (for increasing number of traces)
        true_kb_ranks = np.array([kbs.index(true_key_bytes_batch)
                                  for kbs in final_rankings]) # n_traces x n_traces

        #avg_true_kb_ranks = np.array([np.mean(true_kb_ranks[:][m])
        #                              for m in range (stop - start)]) # 1 x n_traces

        ranks_per_exp.append(true_kb_ranks)

    # After the experiments, average the ranks
    ranks_per_exp = np.vstack(ranks_per_exp) # n_exp x n_traces
    ge = np.mean(ranks_per_exp, axis=0) # 1 x n_traces

    return ge


def retrieve_key_byte(preds, pltxt_bytes, target):

    """
    Retrieves the most probable key-byte for increasing number of attack traces,
    given some predictions.

    Parameters:
        - preds (np.ndarray):
            Model predictions.
        - pltxt_bytes (np.array):
            Plaintext used during the encryption (single byte).
        - target (str):
            Target of the attack.

    Returns:
        - resulting_key_bytes (np.array):
            Most probable key-byte for increasing number of attack traces.
    """

    # # Consider all couples predictions-plaintext
    # all_preds_pltxt = list(zip(preds, pltxt_bytes))

    # # Sample randomly the predictions that will generate the final result
    # sampled = random.sample(all_preds_pltxt, n_traces)
    # sampled_preds, sampled_pltxt_bytes = list(zip(*sampled))

    # Compute the final rankings (for increasing number of traces)]
    final_rankings = compute_final_rankings(preds, pltxt_bytes, target)

    resulting_key_bytes = np.array([ranking[0] for ranking in final_rankings])

    return resulting_key_bytes


def min_att_tr(ge, threshold=0.5):

    """
    Computes the minimum number of attack traces that allows to have Guessing
    Entropy values less that a given threshold.

    Parameters:
        - ge (np.array):
            Guessing Entropy to consider.
        - threshold (float, default=0.5):
            Threshold for GE values.

    Returns:
        - min_att_traces (int):
            Minimum number of attack traces to have GE values less than the
            threshold.
    """

    min_att_traces = ge.shape[0] # ge is a np.array with (N,) as shape

    for i, el in enumerate(ge):
        if el <= threshold:
            min_att_traces = i + 1 # +1 because the actual number of traces is index+1
            break

    return min_att_traces