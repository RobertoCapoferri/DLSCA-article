import trsfile
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

import aes


class DataGenerator(Sequence):
    
    def __init__(self, tr_file, tr_indices, batch_size, target, n_classes, byte_idx, scaler, to_fit=True, shuffle_on_epoch_end=True, cnn=False):
        self.tr_file = tr_file
        self.tr_indices = tr_indices # "Rows" to be read from the .TRS file. len(tr_indices) is the total number of traces to collect.
        self.batch_size = batch_size
        self.target = target
        self.n_classes = n_classes
        self.byte_idx = byte_idx
        self.scaler = scaler
        self.to_fit = to_fit
        self.shuffle_on_epoch_end = shuffle_on_epoch_end
        self.cnn = cnn
        
        self.on_epoch_end()
        
        
    def __len__(self):
        # Generates the number of batches per epoch
        return int(np.floor(len(self.tr_indices) / self.batch_size))
        
        
    def __getitem__(self, index):
        # Generates a batch of data
        i = index * self.batch_size
        batch_indices = self.tr_indices[i : i+self.batch_size]
        
        x, y = self._read_traces(batch_indices)
        
        if self.cnn:
            x = x.reshape(x.shape[0], x.shape[1], 1)
        
        if self.to_fit:
            return x, y
        else:
            return x
    
    
    def on_epoch_end(self):
        # Eventually shuffles the order of the traces
        if self.shuffle_on_epoch_end:
            np.random.shuffle(list(self.tr_indices))
    
    
    def _read_traces(self, batch_indices):
        # Reads the traces from file
        x = []
        y = []
        
        with trsfile.open(self.tr_file, 'r') as traces:
            batch_traces = [traces[i] for i in batch_indices]
            for tr in batch_traces:
                s = tr.samples
                l = self._retrieve_label(tr)
                
                x.append(s)
                y.append(l)
                
        x = np.vstack(x)
        x = self.scaler.transform(x)
        y = np.vstack(y)
        
        return x, y

# #         x = []
# #         y = []

# #         with ExitStack() as stack:
    
# #             tracesets = [stack.enter_context(trsfile.open(tr_file, 'r')) for tr_file in self.tr_files]
        
            
            
# #             batch_traces = [tset[i] 
# #                             for i in batch_indices
# #                             for tset in tracesets]
# #             random.shuffle(batch_traces)

# #             # count = 0
# #             # for i, tr_tuple in enumerate(zip(*tracesets)):
            
# #             # for tr_tuple in batch_tr_tuples:
# #             #     # if i in batch_indices:
# #             #         # if count < self.tr_per_file:
# #             #     x.append([tr.samples for tr in tr_tuple])
# #             #     y.append([self._retrieve_label(tr) for tr in tr_tuple])
# #             #             # count += 1
# #             #         # else:
# #             #             # break
            
# #             for tr in batch_traces:
# #                 samples = tr.samples
# #                 # samples = (samples-np.min(samples)) / (np.max(samples)-np.min(samples))
# #                 x.append(samples)
# #                 y.append(self._retrieve_label(tr))

# #         x = np.vstack(x)
# #         y = np.vstack(y)
        
#         return x, y
        
        
    def _retrieve_label(self, tr):
        # Retrieves the label associated to the given trace
        p = np.array(tr.get_input())
        k = np.array(tr.get_key())
        l = aes.labels_from_key(p, k, self.target)
        l = l[self.byte_idx]
        
        l = to_categorical(l, self.n_classes)
        
        return l