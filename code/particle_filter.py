import numpy as np
import matplotlib.pyplot as plt
import sys
from load_data import load
from data_test import *
import math
from pr2_utils import *

def first_scan(lidar, map, res, xmin, xmax, ymin, ymax):
    """
    First scan of the particle filter
    """
    scan = lidar["range"][:,0]
    ranges = scan
    angles = []
    angles.append(lidar["angle_min"])
    for i in range(ranges.shape[0] - 1):
        angles.append(angles[-1] + lidar["angle_increment"][0][0])
    angles = np.array(angles)
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]

    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)

    x_ranges = np.ceil((x - xmin) / res ).astype(np.int16)-1
    y_ranges = np.ceil((y-0.3 - ymin) / res ).astype(np.int16)-1

    x_map = np.ceil((0 - xmin) / res ).astype(np.int16)-1
    y_map = np.ceil((0 - ymin) / res ).astype(np.int16)-1
    
    for i in range(x_ranges.shape[0]):
        x_occ, y_occ = (bresenham2D(x_map, y_map, x_ranges[i], y_ranges[i]))
        for m in range(x_occ.shape[0]):
                if m == (x_occ.shape[0] - 1):
                    map[int(x_occ[m]), int(y_occ[m])] -= math.log(9)
                else:
                    map[int(x_occ[m]), int(y_occ[m])] += math.log(9)
                if map[int(x_occ[m]), int(y_occ[m])] < -10:
                    map[int(x_occ[m]), int(y_occ[m])] = -10
                if map[int(x_occ[m]), int(y_occ[m])] > 6:
                    map[int(x_occ[m]), int(y_occ[m])] = 6
    
    return map

def update_step(t, lidar, particles, weights, map, res, xmin, xmax, ymin, ymax, N):
    """
    Update step of the particle filter
    """

    scan = lidar["range"][:,t]
    ranges = scan
    angles = []
    angles.append(lidar["angle_min"])
    for i in range(ranges.shape[0] - 1):
        angles.append(angles[-1] + lidar["angle_increment"][0][0])
    angles = np.array(angles)
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]

    x_lidar = ranges * np.cos(angles)
    y_lidar = ranges * np.sin(angles)

    lidar_coord = np.array([x_lidar, y_lidar])

    theta = particles[: , 2]
    
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    R = np.transpose(R, (2, 0, 1))
    t = np.array([[particles[:, 0]], [particles[:, 1] + 0.3]])
    t = np.transpose(t, (2, 0, 1))

    map_coord_N = R @ lidar_coord + t
    corr = np.zeros(N)

    for j in range(N):
        map_coord_N[j, :, :] = np.ceil((map_coord_N[j, :, :] - np.array([[xmin], [ymin]])) / res ).astype(np.int16)-1
        temp_map = map_coord_N[j, :, :]
        for i in range(temp_map.shape[1]):
            if map[int(temp_map[0, i]), int(temp_map[1, i])] < 0:
                corr[j] += abs(map[int(temp_map[0, i]), int(temp_map[1, i])])
    
    weights = weights * corr
    weights = weights / np.sum(weights)
    
    update_id = np.argmax(weights)
    
    map_coord = map_coord_N[update_id, :, :]

    x = particles[update_id, 0]
    y = particles[update_id, 1]

    x = np.ceil((x - xmin) / res ).astype(np.int16)-1
    y = np.ceil((y - ymin) / res ).astype(np.int16)-1

    for l in range(map_coord.shape[1]):
        x_occ, y_occ = bresenham2D(x, y, map_coord[0, l], map_coord[1, l])
        for m in range(x_occ.shape[0]):
            if m == (x_occ.shape[0] - 1):
                map[int(x_occ[m]), int(y_occ[m])] -= math.log(9)
            else:
                map[int(x_occ[m]), int(y_occ[m])] += math.log(9)
            if map[int(x_occ[m]), int(y_occ[m])] < -10:
                map[int(x_occ[m]), int(y_occ[m])] = -10
            if map[int(x_occ[m]), int(y_occ[m])] > 6:
                map[int(x_occ[m]), int(y_occ[m])] = 6

    return weights, map


def predict_step(v, w, particles):
    """
    Particle filter for localization
    """
    noise = np.random.normal(0, 5, (particles.shape[0], 2))
    noise_w = np.random.normal(0, 0.5, (particles.shape[0], 1))
    angles = particles[:, 2]
    particles[:,0] = particles[:,0] + ((v * np.cos(angles)) + noise[:,0]) * 0.025
    particles[:,1] = particles[:,1] + ((v * np.cos(angles)) + noise[:,1]) * 0.025
    particles[:,2] = particles[:,2] + (w + noise_w[:,0]) * 0.025
        
    return particles

def particle_filter(lidar, v, w, particles, weights, map, res, xmin, xmax, ymin, ymax, N):
    """
    Particle filter for localization
    """
    
    for i in range(v.shape[0]):

        start = time.time()

        particles = predict_step(v[i], w[i], particles)
        weights, map = update_step(i, lidar, particles, weights, map, res, xmin, xmax, ymin, ymax, N)

        print("Weights: ", weights)
        print(i)
        print("Time: ", time.time() - start)
        print("=================================")
        if i==1000:
            print((max(v), max(w)))
            print((min(v), min(w)))
            break

    return particles, weights, map

def dead_reckon (lidar, v, w, map, res, xmin, xmax, ymin, ymax):
    """
    Dead reckoning for localization
    """
    pose = [0, 0, 0]
    for i in range(v.shape[0]):
        start = time.time()
        scan = lidar["range"][:,i]
        ranges = scan
        angles = []
        angles.append(lidar["angle_min"])
        for j in range(ranges.shape[0] - 1):
            angles.append(angles[-1] + lidar["angle_increment"][0][0])
        angles = np.array(angles)
        indValid = np.logical_and((ranges < 30),(ranges> 0.1))
        ranges = ranges[indValid]
        angles = angles[indValid]
        x_lidar = ranges * np.cos(angles)
        y_lidar = ranges * np.sin(angles)

        x = pose[0]
        y = pose[1]

        theta = pose[2]
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t = np.array([[x], [y + 0.3]])

        lidar_coord = np.vstack((x_lidar, y_lidar))
        map_coord = (R @ lidar_coord) + t

        #map_coord[0,:] = np.ceil((map_coord[0,:] - xmin) / res ).astype(np.int16)-1
        #map_coord[1,:] = np.ceil((map_coord[1,:] - ymin) / res ).astype(np.int16)-1
        map_coord[:,:] = np.ceil((map_coord[:,:] - np.array([[xmin], [ymin]])) / res ).astype(np.int16)-1

        x_map = np.ceil((x - xmin) / res ).astype(np.int16)-1
        y_map = np.ceil((y - ymin) / res ).astype(np.int16)-1

        for l in range(map_coord.shape[1]):
            x_occ, y_occ = bresenham2D(x_map, y_map, map_coord[0, l], map_coord[1, l])
            for m in range(x_occ.shape[0]):
                if m == (x_occ.shape[0] - 1):
                    map[int(x_occ[m]), int(y_occ[m])] -= math.log(4)
                else:
                    map[int(x_occ[m]), int(y_occ[m])] += math.log(4)
                if map[int(x_occ[m]), int(y_occ[m])] < -7:
                    map[int(x_occ[m]), int(y_occ[m])] = -7
                if map[int(x_occ[m]), int(y_occ[m])] > 4:
                    map[int(x_occ[m]), int(y_occ[m])] = 4

        
        pose[0] += v[i] * np.cos(pose[2]) * 0.025
        pose[1] += v[i] * np.sin(pose[2]) * 0.025
        pose[2] += w[i] * 0.025
        
        print("Dead Reckoning: ", pose)
        print(i)
        print("Time: ", time.time() - start)
        print("=================================")
        if i==1000:
            break
    
    return map