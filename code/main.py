import numpy as np
import matplotlib.pyplot as plt
import sys
from load_data import load
from data_test import *
import math
from pr2_utils import *
from particle_filter import *

if __name__ == "__main__":

    dataset = sys.argv[1]
    encoder, lidar, imu, kinect = load(int(dataset))
    
    x_enc, y_enc, theta_enc, v_enc, w_enc = encoder_test(encoder)
    x_imu, y_imu, theta_imu, v_imu, w_imu = imu_test(imu)

    v, w = synchronize_data(encoder, imu, v_enc, w_imu)

    res = 0.1
    xmin = -30   
    xmax = 30
    ymin = -30
    ymax = 30
    sizex = int(np.ceil((xmax - xmin) / res + 1))
    sizey = int(np.ceil((ymax - ymin) / res + 1))
    map = np.zeros((sizex, sizey), dtype = np.int8)

    N = 10

    particles = np.zeros((N, 3))
    weights = np.ones(N) / N

    #x_temp, y_temp, theta_temp, states = motion_model(v, w)

    map = first_scan(lidar, map, res, xmin, xmax, ymin, ymax)

    #plt.plot(x_temp, y_temp)
    #plt.show()

    map_temp = dead_reckon(lidar, v, w, map, res, xmin, xmax, ymin, ymax)
    plt.imshow(map_temp, cmap="gray")
    plt.show()

    #occupancy_map = particle_filter(lidar, v, w, particles, weights, map, res, xmin, xmax, ymin, ymax, N)    