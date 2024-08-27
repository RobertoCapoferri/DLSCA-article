import gc
import trsfile
import numpy as np

from tqdm import tqdm

class BatchStandardScaler():

    def __init__(self, trace_file, tot_traces, trace_len, n_batch, train_size=1.0):

        self.trace_file = trace_file
        self.tot_traces = tot_traces
        self.trace_len = trace_len
        self.n_batch = n_batch
        self.train_size = train_size

        self.batch_size = int(tot_traces / n_batch)

        self.mean = np.zeros((trace_len, ))
        self.var = np.zeros((trace_len, ))


    def _online_mean_std(self, traces):

        for i in tqdm(range(self.n_batch)):
            # Set previous and current number of traces
            m = i * self.batch_size
            n = self.batch_size
            
            # Get current batch of data
            new_x = [tr.samples for tr in traces[m:m+n]]
            new_x = np.vstack(new_x)
            
            # Consider old mean and compute the current one
            old_mean = self.mean
            new_mean = np.mean(new_x, axis=0)
            
            # Consider old var and compute the current one
            old_var = self.var
            new_var = np.var(new_x, axis=0)
            
            # Update mean and var
            self.mean = (m*old_mean + n*new_mean) / (m+n)
            var_m1 = (m*old_var + n*new_var) / (m+n)
            var_m2 = m * n / (m+n)**2 * (old_mean-new_mean)**2 # Precedence: (), **, *, /
            self.var = var_m1 + var_m2

            # Manually free memory
            del new_x
            gc.collect()
            
        # Compute std
        self.std = np.sqrt(self.var)

    
    def fit(self):

        with trsfile.open(self.trace_file, 'r') as traces:
            
            self._online_mean_std(traces)


    def transform(self, data):

        scaled_data = (data - self.mean) / self.std

        return scaled_data


