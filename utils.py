import numpy as np
import logging
import os
import json
import sys

log = logging.getLogger()
log.setLevel(logging.DEBUG)

def create_file_name(region, isotope, v_bin):
    return "G%s-%sCO_%s.fits"%(str(region), str(isotope), str(v_bin))

def extract_patches(input_map, target_map, length, v_stride, h_stride):
    assert input_map.shape == target_map.shape
    height, width = input_map.shape
    gap = int((length-1)/2.)
    triples = []
    for j in range(gap, width-gap, h_stride):
        for i in range(gap, height-gap, v_stride):
            target = target_map[i, j]
            v_patch = input_map[i, j-gap:j+gap+1]
            h_patch = input_map[i-gap:i+gap+1, j]
            triples.append((h_patch, v_patch, target))
    return triples

def reconstruct_sky_map(predictions, info_file, v_bin):
    with open(info_file, 'r') as info:
        info = json.load(info)
        sorted_regions = sorted(info.items(), key=lambda x: x[1][2])
        start = 0
        for item in sorted_regions:
            h, w = item[1][:2]
            length = h*w
            order = item[1][-1]-1
            region_pixels = predictions[start: start+length]
            region = region_pixels.reshape(h, w)
            np.save("%s_smart_map_%d.npy"%(item[0], v_bin), region)
            start = start+length

        
