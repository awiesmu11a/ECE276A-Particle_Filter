import numpy as np
import matplotlib.pyplot as plt
import sys
from load_data import load
from data_test import *
import math
from pr2_utils import *

if __name__ == "__main__":
    dataset = sys.argv[1]
    encoder, lidar, imu, kinect = load(int(dataset))

    

    
    
    
    
    freq_match = 3

    x_enc, y_enc, theta_enc, v_enc, w_enc = encoder_test(encoder)
    x_imu, y_imu, theta_imu, v_imu, w_imu = imu_test(imu)
    
    v = []
    w = []
    j = 0
    tsimu = []
    tsenc = []

    for i in range(w_imu.shape[0]):
        if abs(imu["timestamps"][i] - encoder["timestamps"][j]) <= 0.01:
            w.append(w_imu[i])
            v.append(v_enc[j])
            j += 1

        if encoder["timestamps"][j] < imu["timestamps"][i]:
            j += 1
    
    w = np.array(w)
    v = np.array(v)

    map = np.ones((70, 70))
    map_2 = map

    x, y, theta, states = motion_model(v, w)
    x_lidar, y_lidar, map_1 = lidar_test_path(lidar, x, y, theta, 
        freq_match, reshape=False)
    x, y, map_1 = lidar_occupancy(lidar, states, map)

    
    br = np.zeros((1,2))

    for i in range(x_lidar[0].shape[0]):
        temp = (bresenham2D(0, 0, x_lidar[0][i], y_lidar[0][i]))
        br = np.concatenate((br, temp.T), axis=0)
    #map = np.ones((20, 20))
    br = br[1:]
    for i in range(br.shape[0]):
        map_2[34 - int(br[i][1]), 34 - int(br[i][0])] = 0
    
    plt.imshow(map_2, cmap="gray")
    #plt.savefig("map.png")
    #plt.scatter(x_lidar[::50], y_lidar[::50], s=0.1)
    plt.scatter(34 - x_lidar[0], 34 - y_lidar[0], color="red", s=0.1)
    plt.savefig("path.png")