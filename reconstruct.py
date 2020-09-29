import numpy as np
import logging
import os
import json
import sys
from utils import *

log = logging.getLogger()
log.setLevel(logging.DEBUG)

def main(predictions_dir, info_file):
    for file_name in os.listdir(predictions_dir):
        if file_name[-4:] == ".npy":
            v_bin = file_name[:-4].split("_")[1]
            v_bin = int(v_bin[2:])
            predictions = np.load(os.path.join(predictions_dir, file_name))
            reconstruct_sky_map(predictions, info_file, v_bin)




if __name__ == "__main__":
    predictions_dir = sys.argv[1]
    info_file = sys.argv[2]
    main(predictions_dir, info_file)
