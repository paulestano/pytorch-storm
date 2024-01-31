# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from scipy import stats
import torch
import torch.nn.functional as F

# Use a Leaky Bucket to store a fraction of most recent statistics 
# Wikepedia article: https://en.wikipedia.org/wiki/Leaky_bucket
class LeakyBucket(object):
    def __init__(self, size, ratio, dtype, device, fixed_len=-1):
        '''
        size:  size of allocated memory buffer to keep the leaky bucket queue,
               which will be doubled whenever the memory is full
        ratio: integer ratio of total number of samples to numbers to be kept:
               1 - keep all, 
               2 - keep most recent 1/2, 
               3 - keep most recent 1/3,
               ... 
        fixed_len: fixed length to keep, ratio >=1 becomes irrelevant
        '''
        self.size = size
        self.ratio = int(ratio)
        self.fixed_len = int(fixed_len)

        self.buffer = torch.zeros(size, dtype=dtype, device=device)
        self.count = 0          # number of elements kept in queue (excluding leaked)
        self.start = 0          # count = end - start
        self.end = 0
        self.total_count = 0    # total number of elements added (including leaked)
 
    def reset(self):
        self.buffer.zero_()    
        self.count = 0          
        self.start = 0
        self.end = 0
        self.total_count = 0

    def double_size(self):
        self.size *= 2
        self.buffer.resize_(self.size)

    def add(self, val):
        if self.end == self.size:               # when the end index reach size
            self.double_size()                      # double the size of buffer

        self.buffer[self.end] = val             # always put new value at the end
        self.end += 1                           # and increase end index by one

        if self.fixed_len > 0:
            if self.count == self.fixed_len:
                self.start += 1
            else:
                self.count += 1
        else:
            if self.total_count % self.ratio == 0:  # if leaky_count is multiple of ratio
                self.count += 1                         # increase count in queue by one
            else:                                   # otherwise leak and keep same count
                self.start += 1                         # increase start index by one

        self.total_count += 1                   # always increase total_count by one

        # reset start index to 0 and end index to count to save space
        if self.start >= self.count:
            self.buffer[0:self.count] = self.buffer[self.start:self.end]
            self.start = 0
            self.end = self.count

    # ! Need to add safeguard to allow compute only if there are enough entries
    def mean_std(self, mode='bm'):
        mean = torch.mean(self.buffer[self.start:self.end]).item()

        if mode == 'bm':        # batch mean variance
            b_n = int(math.floor(math.sqrt(self.count)))
            Yks = F.avg_pool1d(self.buffer[self.start:self.end].unsqueeze(0).unsqueeze(0), kernel_size=b_n, stride=b_n).view(-1)
            diffs = Yks - mean
            std = math.sqrt(b_n /(len(Yks)-1))*torch.norm(diffs).item()
            dof = b_n - 1
        elif mode == 'olbm':    # overlapping batch mean
            b_n = int(math.floor(math.sqrt(self.count)))
            Yks = F.avg_pool1d(self.buffer[self.start:self.end].unsqueeze(0).unsqueeze(0), kernel_size=b_n, stride=1).view(-1)
            diffs = Yks - mean
            std = math.sqrt(b_n*self.count/(len(Yks)*(len(Yks)-1)))*torch.norm(diffs).item()
            dof = self.count - b_n
        else:                   # otherwise use mode == 'iid'
            std = torch.std(self.buffer[self.start:self.end]).item()
            dof = self.count - 1

        return mean, std, dof