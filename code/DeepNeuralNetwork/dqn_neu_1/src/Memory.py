#!/usr/bin/env python

import numpy as np
from numpy import random

import random

class Memory:
  # constructor
  def __init__(self, memory_size):
    self.size = memory_size
    self.memory = []
    self.store_cnt = 0

  # store experience as a dictionary inside a list at a random index
  def store_experience(self, resulting_state, last_state, reward,
                       action, done):
    tuple = {"resulting_state":resulting_state,
             "last_state":last_state, "reward":reward,
             "action":action, "done":done}
    if (self.store_cnt <= self.size):
      self.memory.append(tuple)
      self.store_cnt += 1
    else:
      i = np.random.randint(0, self.size)
      self.memory[i] = tuple
      random.shuffle(self.memory)
      self.store_cnt += 1

  # get a list of size batchsize with memory samples
  def get_memory_batch(self, batchsize):
    return_value = []
    if(self.store_cnt >= self.size):
      index = np.random.randint(0, self.size, batchsize)
    else:
      index = np.random.randint(0, self.store_cnt, batchsize)
    for i in range(batchsize):
      return_value.append(self.memory[index[i]])
    return return_value