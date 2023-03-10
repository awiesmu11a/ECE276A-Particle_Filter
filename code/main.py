import numpy as np
import matplotlib.pyplot as plt
import sys
from load_data import load
from data_test import *
from pr2_utils import *
from particle_filter import *
from texture_map import *
import cv2

if __name__ == "__main__":

    dataset = sys.argv[1]
    encoder, lidar, imu, kinect = load(int(dataset))

    path_rgb = "./../../data/dataRGBD/RGB" + dataset + "/"
    path_depth = "./../../data/dataRGBD/Disparity" + dataset + "/"

    x_enc, y_enc, theta_enc, v_enc, w_enc = encoder_test(encoder)
    x_imu, y_imu, theta_imu, v_imu, w_imu = imu_test(imu)

    v, w, ts_motion = synchronize_data(encoder, imu, v_enc, w_imu)
    lidar_ranges, v_sync, w_sync, ts_occ = synchronize_lidar(lidar, v, w, ts_motion)
    lidar_ranges = lidar_ranges.T

    x, y, theta, states = motion_model(v_sync, w_sync, 0.025) # 0.025 is the timestep
    fig_trajectory = plt.figure()
    plt.plot(x, y)
    #Save the trajectory
    plt.savefig("trajectory_1.png")

    x_visualize = []
    y_visualize = []
    for j in range(10):
        v_temp = v_sync + np.random.normal(0, (0.2 * abs(np.max(v_sync))), (v_sync.shape[0]))
        w_temp = w_sync + np.random.normal(0, (0.3 * abs(np.max(w_sync))), (w_sync.shape[0]))
        x_temp, y_temp, theta_temp, states_temp = motion_model(v_temp, w_temp, 0.025)
        x_visualize.append(x_temp)
        y_visualize.append(y_temp)

    fig_visulize = plt.figure()
    for i in range(len(x_visualize)):
        plt.plot(x_visualize[i], y_visualize[i])
    #Save the visualization
    plt.savefig("visualization_1.png")

    res = 0.1
    xmin = -40
    xmax = 40
    ymin = -40
    ymax = 40
    sizex = int(np.ceil((xmax - xmin) / res + 1))
    sizey = int(np.ceil((ymax - ymin) / res + 1))
    map = np.zeros((sizex, sizey), dtype = np.float64)
    
    N = 40

    x_map = np.ceil((x - xmin) / res ).astype(np.int16)-1
    y_map = np.ceil((y - ymin) / res ).astype(np.int16)-1


    particles = np.zeros((N, 3))
    weights = np.ones(N) / N

    map = first_scan(lidar_ranges, map, res, xmin, xmax, ymin, ymax)

    fig_first_scan = plt.figure()    
    plt.imshow(map, cmap="gray")
    #Save the first scan
    plt.savefig("first_scan_1.png")
    map_temp = dead_reckon(lidar_ranges, v_sync, w_sync, map, res, xmin, xmax, ymin, ymax) #Generate dead reckoning map
    fig_dead_reckon = plt.figure()
    plt.imshow(map_temp, cmap="gray")
    #Save the dead reckoning map
    plt.savefig("dead_reckon_1.png")

    particles, weights, occupancy_map, = particle_filter(lidar_ranges, v_sync, w_sync, particles, weights, map, res, xmin, xmax, ymin, ymax, N)

    texture = texture_map_plot(dataset, kinect["rgb_timestamps"], kinect["disparity_timestamps"], map, res, xmin, xmax, ymin, ymax, x, y, theta, path_rgb, path_depth, ts_occ)
    