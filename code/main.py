import numpy as np
import matplotlib.pyplot as plt
import sys
from load_data import load
from data_test import *

if __name__ == "__main__":
    dataset = sys.argv[1]
    encoder, lidar, imu, kinect = load(int(dataset))

    freq_match = 3

    x_enc, y_enc, theta_enc, v_enc, w_enc = encoder_test(encoder)
    x_imu, y_imu, theta_imu, v_imu, w_imu = imu_test(imu)
    x_lidar, y_lidar = lidar_test_path(lidar, x_imu, y_imu, theta_imu, 
        freq_match, reshape=True)

    plt.scatter(x_lidar[::200], y_lidar[::200], s=0.1)
    #plt.plot(x_imu[:], y_imu[:], 'r')
    plt.show()




    """
    for j in range(lidar["range"].shape[1]):
        x = []
        y = []
        theta = lidar["angle_min"]
        for i in range(lidar["range"].shape[0]):
            x.append(lidar["range"][i, j] * np.cos(theta))
            y.append(lidar["range"][i, j] * np.sin(theta))
            theta += lidar["angle_increment"][0][0]
        x = np.array(x)
        y = np.array(y)
        plt.figure()
        plt.plot(x, y, 'o')
        plt.savefig("./../data/lidar_" + str(j) + ".png")
        plt.close()
    """
        

    
        

    

