import numpy as np
import pandas as pd

def load_data(dataset, num_compressed):
  #load data
  original_signal = pd.read_csv('./Data/minusmean'+dataset+'.csv', header=None)
  original_signal = original_signal.as_matrix()
  original_signal = np.transpose(original_signal)
  n_rec = np.size(original_signal,0)
  p_rec = np.size(original_signal,1)
  print ('n_rec, p_rec',n_rec, p_rec)

  # load sensor locations computed by Algorithm 1
  sensors = pd.read_csv('./Data/'+dataset+'_sensors_'+str(num_compressed)+'.csv', header=None)
  print ('sensors shape: ', sensors.shape)
  sensors = sensors.as_matrix()
  sensors = np.reshape(sensors,[num_compressed])
  print('reading data done!')
  return original_signal, sensors, n_rec, p_rec
