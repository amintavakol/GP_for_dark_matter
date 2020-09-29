import numpy as np
from astropy.io import fits
import logging
from pathlib import Path
import os
from typing import Tuple
import h5py
import json
import sys

from utils import *

log = logging.getLogger()
log.setLevel(logging.DEBUG)

#V_BINS = 20
V_BINS = 17

def create_data(v_bin, h_params, out_filename):
    #TRAIN_REGIONS = h_params['train_regions']
    #TEST_REGIONS = h_params['test_regions']
    
    #CO12_PATH = h_params['co12_path']
    #CO13_PATH = h_params['co13_path']
    DATA_PATH = h_params['data_path']
    PATH_TO_SAVE = h_params['path_to_save']

    f = h5py.File(os.path.join(PATH_TO_SAVE, "TRAIN_VB"+str(v_bin)+"_"+out_filename), 'w')
    g = h5py.File(os.path.join(PATH_TO_SAVE, "TEST_VB"+str(v_bin)+"_"+out_filename), 'w')
    test_regions_dims = {} # need to store the dims of each region to be used later for reconstruction

    verticals = []
    horizontals = []
    targets = []
    
    train_co12 = h5py.File(os.path.join(DATA_PATH, "training_co12.h5"))
    train_co13 = h5py.File(os.path.join(DATA_PATH, "training_co13.h5"))
    test_co12 = h5py.File(os.path.join(DATA_PATH, "testing_co12.h5"))
    test_co13 = h5py.File(os.path.join(DATA_PATH, "testing_co13.h5"))
    
    TRAIN_REGIONS = train_co12.keys()
    TEST_REGIONS = test_co12.keys()


    log.info("creating train data for velocity bin: %d"%v_bin)
    for region in TRAIN_REGIONS:
        
        #img_data_co12 = fits.getdata(os.path.join(CO12_PATH, filename))
        #img_data_co13 = fits.getdata(os.path.join(CO13_PATH, filename))

        img_data_co12 = np.array(train_co12[region])    
        img_data_co13 = np.array(train_co13[region])    

        input_map = img_data_co12[v_bin, 5:-5, 5:-5]
        target_map = img_data_co13[v_bin, 5:-5, 5:-5]

        triples = extract_patches(input_map, target_map, h_params['length'], h_params['v_stride'], h_params['h_stride'])
        log.info("%d triples are extracted from region %s"%(len(triples), region))
        
        for triple in triples:
            horizontals.append(triple[0])
            verticals.append(triple[1])
            targets.append(triple[2])

    f.create_dataset("verticals", data=np.array(verticals))
    f.create_dataset("horizontals", data=np.array(horizontals))
    f.create_dataset("targets", data=np.array(targets))

    log.info("train data are imported into hdf5 file")
    f.close()
    
    verticals = []
    horizontals = []
    targets = []
    
    log.info("creating test data for velocity bin: %d"%v_bin)
    region_order = 0 # neede for reconstruction of test maps
    for region in TEST_REGIONS:
        
        #img_data_co12 = fits.getdata(os.path.join(CO12_PATH, filename))
        #img_data_co13 = fits.getdata(os.path.join(CO13_PATH, filename))

        img_data_co12 = np.array(test_co12[region])    
        img_data_co13 = np.array(test_co13[region])    

        input_map = img_data_co12[v_bin, 5:-5, 5:-5]
        target_map = img_data_co13[v_bin, 5:-5, 5:-5]

        #triples = extract_patches(input_map, target_map, h_params['length'], h_params['v_stride'], h_params['h_stride'])
        triples = extract_patches(input_map, target_map, h_params['length'], 1, 1) # need stride size 1 to be able to reconstruct the map
        
        h, w = input_map.shape
        test_h = h-(h_params['length']-1)
        test_w = w-(h_params['length']-1)
        assert len(triples) == test_h*test_w
        

        log.info("%d triples are extracted from region %s"%(len(triples), region))
        
        for triple in triples:
            horizontals.append(triple[0])
            verticals.append(triple[1])
            targets.append(triple[2])
        
        region_order += 1
        test_regions_dims[region] = (test_h, test_w, region_order)
    
    g.create_dataset("verticals", data=np.array(verticals))
    g.create_dataset("horizontals", data=np.array(horizontals))
    g.create_dataset("targets", data=np.array(targets))

    log.info("test data are imported into hdf5 file")
    g.close()
    
    with open("info_test_regions.json", 'w') as h:
        json.dump(test_regions_dims, h)
        
if __name__ == "__main__":
    out_filename = sys.argv[1]
    h_params_file = sys.argv[2]
    f = open(h_params_file)
    h_params = json.load(f)
    for i in range(V_BINS):
        create_data(i, h_params, out_filename)
