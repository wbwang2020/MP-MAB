# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:36:37 2019

@author: wenbo2017
"""

#import multiprocessing as mp
#import time
#
#class someClass(object):
#
#  def __init__(self):
#      self.var = 1
#
#  def test(self):
#      print(self)
#      print ("Variable value: {}".format(self.var))
#      self.var += 1
#      
#  def apply_async_with_callback(self):
#      pool = mp.Pool(processes = 3)
#      for i in range(10):
#          pool.apply_async(self.test) #, callback = self.log_result
#         
#      pool.close()
#      pool.join()
#
#
#if __name__ == '__main__':
#    sc = someClass()
#    
#    sc.apply_async_with_callback()
    

#from multiprocessing import Pool
#import time
#from tqdm import *
#
#def _foo(my_number):
#   square = my_number * my_number
#   time.sleep(1)
#   return square 
#
#if __name__ == '__main__':
#    with Pool(processes=2) as p:
#        max_ = 30
#        with tqdm(total=max_) as pbar:
#            for i, _ in tqdm(enumerate(p.imap_unordered(_foo, range(0, max_)))):
#                pbar.update()


import numpy as np

import multiprocessing as mp
from tqdm import tqdm
from time import sleep

SENTINEL = 1

def test(q=None):
    for i in range(1000):
        sleep(0.01)
        
        if q is not None:
            q.put(SENTINEL)

def listener(q, nbProcess):
    pbar = tqdm(total = 1000*nbProcess)
    for item in iter(q.get, None):
        pbar.update()

if __name__ == '__main__':
#    pool = mp.Pool(processes=5)
#    manager = mp.Manager()
#    queue = manager.Queue()
#    
#    proc = mp.Process(target=listener, args=(queue, 5))
#    
#    for ii in range(5):
#        pool.apply_async(test, args=(queue, ))
#        
#    proc.start()
#    pool.close()
#    pool.join()
#    queue.put(None)
#    proc.join()
#    
#    print("process is done")
#    c = np.array([])
    c = None
    
    d = np.array([1, 2, 0, 4, 0])
    
    idx = np.where(d != 0)
    
    d[idx] = -1
    
    print(d)
    
    
#    a = [0, 0, 0, 0, 0]
#    
#    arm_selected = np.nonzero(a)
#    
#    print(arm_selected[0])
##    print(arm_selected[1])
#    
#    indx = np.where(a == 6)
#    
#    print(indx)
#    print(indx[0].ndim)
#    print(indx[0].shape)
#    
#    aa = np.array(a)
#    
#    aa[:] = 0
#    
#    b = np.array(list(range(0, 10)))
#    print(b)
#    
#    print(aa)
#    q = mp.Queue()
#    proc = mp.Process(target=listener, args=(q,))
#    proc.start()
#    workers = [mp.Process(target=test, args=(q,)) for i in range(5)]
#    for worker in workers:
#        worker.start()
#    for worker in workers:
#        worker.join()
#    q.put(None)
#    proc.join()