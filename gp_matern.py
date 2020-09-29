import numpy as np
import h5py
import json
import sys
import os

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel as W
from sklearn.gaussian_process.kernels import Matern as M
from sklearn import preprocessing

from spinner import Spinner

import multiprocessing as mp
from multiprocessing import Pool, freeze_support

import logging as log
log.basicConfig(level=log.INFO)

def load_train_data(filename, vertical=True):
    f = h5py.File(filename, 'r')
    targets = np.array(f['targets'])
    if vertical:
        x_data = np.array(f['verticals'])
        assert x_data.shape[0] == targets.shape[0]
        return x_data, targets
    
    else:
        x_data = np.array(f['horizontals'])
        assert x_data.shape[0] == targets.shape[0]
        return x_data, targets


def load_test_data(filename, vertical=True):
    f = h5py.File(filename, 'r')
    targets = np.array(f['targets'])
    if vertical:
        x_data = np.array(f['verticals'])
        assert x_data.shape[0] == targets.shape[0]
        return x_data, targets
    else:
        x_data = np.array(f['horizontals'])
        assert x_data.shape[0] == targets.shape[0]
        return x_data, targets

def preprocess(train_data, test_data):
    scaler = preprocessing.StandardScaler().fit(train_data)
    scaled_train_data = scaler.transform(train_data)
    mean = scaler.mean_
    std = scaler.scale_
    scaled_test_data = scaler.transform(test_data)
    return scaled_train_data, scaled_test_data, mean, std

def prepare_gp():
    white = W(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
    matern = M()
    k = white + matern
    #gp = GaussianProcessRegressor(kernel=k, alpha=1e-3, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5, normalize_y=True, copy_X_train=True, random_state=None)
    gp = GaussianProcessRegressor(kernel=k, alpha=1e-3, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=5, normalize_y=False, copy_X_train=True, random_state=None)

    return gp

def train_and_eval_gp(gp, x_train, y_train, x_test, y_test, v_bin,vertical=True):
    gp.fit(x_train, y_train)
    if vertical:
        print("starting to do predictions for %s patches of velocity bin %d"%("vertical", v_bin))
    else:
        print("starting to do predictions for %s patches of velocity bin %d"%("horizontal", v_bin))
    y_pred = gp.predict(x_test)
    
    mae = np.mean(y_pred - y_test)
    #mae_file.write("mae: %f"%mae)
    #mae_file.write(str(v_bin))
    #mae_file.write("----\n")

    return y_pred, mae
    

def process_v_bin(data_dir, v_bin):
    gp_v = prepare_gp()
    gp_h = prepare_gp()
    
    
    print("start processing velocity bins %d"%v_bin)

    train_file = os.path.join(data_dir, "TRAIN_VB%d_preprocessed_data_GP.hdf5"%v_bin)
    test_file = os.path.join(data_dir, "TEST_VB%d_preprocessed_data_GP.hdf5"%v_bin)
    out_file = os.path.join(data_dir, "OUTPUT_VB%d.npy"%v_bin)


    x_train_v, y_train_v = load_train_data(train_file, vertical=True)
    x_test_v, y_test_v = load_test_data(test_file, vertical=True)
    
    x_train_h, y_train_h = load_train_data(train_file, vertical=False)
    x_test_h, y_test_h = load_train_data(test_file, vertical=False)

    x_train_h, x_test_h, mean_h, std_h = preprocess(x_train_h, x_test_h)
    x_train_v, x_test_v, mean_v, std_v = preprocess(x_train_v, x_test_v)
    
    y_pred_v, mae_v = train_and_eval_gp(gp_v, x_train_v, y_train_v, x_test_v, y_test_v, v_bin, vertical=True)
    y_pred_h, mae_h = train_and_eval_gp(gp_h, x_train_h, y_train_h, x_test_h, y_test_h, v_bin, vertical=False)

    o = (y_pred_v + y_pred_h)/2.

    np.save("mae_error_%d.npy"%v_bin, np.array([mae_v, mae_h]))
    np.save(out_file, o)



def main(data_dir):
    num_bins = 17
    #f = open("mae_file.txt", 'w')
    pool = mp.Pool(num_bins)
    args = [(data_dir, i) for i in range(num_bins)]
    result = pool.starmap(process_v_bin, args)

if __name__ == "__main__":
    _, data_dir = sys.argv
    main(data_dir)

    
