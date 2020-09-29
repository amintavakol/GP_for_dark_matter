import multiprocessing as mp
from multiprocessing import Pool, freeze_support

import sys

def func(a,b):
    print(a*b)

def main(c=3):
    pool = mp.Pool(mp.cpu_count())
    result = pool.starmap(func, [(c,4),(c, 2),(c, 3)])

if __name__ == "__main__":
    freeze_support()
    main(int(sys.argv[1]))

    

