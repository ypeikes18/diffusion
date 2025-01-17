import numpy as np

def get_moving_average(data, window_size):
  return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
