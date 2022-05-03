#!/usr/bin/env python
# coding: utf-8
#
# Simple tensorflow program to compare cpu and gpu execution times.
#

import tensorflow as tf
import numpy as np
import time

def matrix_multiply(size: int)-> dict:
    """
    Time a 2D matrix multiply using tensorflow accross all physical devices.
    
    Args:
     - size (int): The matrix size.
     - returns (dict): {Device string: elapsed time}
     """
    a = np.random.rand(size, size)
    b = np.random.rand(size, size)
    results = {}
    for dev in tf.config.list_physical_devices():
        with tf.device(dev.device_type):
                ta = tf.constant(a)
                tb = tf.constant(b)
                t0 = time.time()
                x = tf.matmul(ta, tb)
                t1 = time.time()
                results.update({dev.device_type:t1 - t0})

   
    return results

print(f'{tf.config.list_physical_devices()}')
print(f'Matrix Multiply Elapsed Time: {matrix_multiply(4096)}')