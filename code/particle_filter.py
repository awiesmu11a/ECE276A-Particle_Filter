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
    y_ranges = np.ceil((y - ymin) / res ).astype(np.int16)-1
    
    for i in range(x_ranges.shape[0]):
        temp = (bresenham2D(0, 0, x_ranges[i], y_ranges[i]))
        map[temp[:,0], temp[:,1]] = 1
    
    return map

def update_step(lidar, particles, weights, map, res, xmin, xmax, ymin, ymax, N):
    """
    Update step of the particle filter
    """

    scan = lidar["ranges"][:,0]
    ranges = scan
    angles = np.arange(lidar["angle_min"], lidar["angle_max"], lidar["angle_increment"])
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    x_lidar = ranges * np.cos(angles)
    y_lidar = ranges * np.sin(angles)

    x_ranges = np.ceil((x_lidar - xmin) / res ).astype(np.int16)-1
    y_ranges = np.ceil((y_lidar - ymin) / res ).astype(np.int16)-1

    for j in range(N):
        particle = particles[j]
        weight = weights[j]
        x = particle[0]
        y = particle[1]
        x_map = np.ceil((x - xmin) / res ).astype(np.int16)-1
        y_map = np.ceil((y - ymin) / res ).astype(np.int16)-1

        theta = particle[2]
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t = np.array([[x], [y - 0.3]])

        lidar_coord = np.vstack((x_ranges, y_ranges))
        map_coord = (R @ lidar_coord) + t
        corr = 0
        for l in range(map_coord.shape[1]):
            x_occ, y_occ = bresenham2D(x, y, map_coord[0, l], map_coord[1, l])
            for m in range(x_occ.shape[0]):
                if map[x_occ[m], y_occ[m]] == 1:
                    corr += 1
        weight = weight * corr
        weights[j] = weight
    
    update_id = np.argmax(weights)
    
    x = particles[update_id, 0]
    y = particles[update_id, 1]
    t = np.array([[x], [y - 0.3]])

    theta = particles[update_id, 2]
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    lidar_coord = np.vstack((x_ranges[i], y_ranges[i]))
    map_coord = (R @ lidar_coord) + t

    for l in range(map_coord.shape[1]):
        x_occ, y_occ = bresenham2D(x, y, map_coord[0, l], map_coord[1, l])
        for m in range(x_occ.shape[0]):
            map[x_occ[m], y_occ[m]] += math.log(4)

    return weights


def predict_step(lidar, v, w, particles, weights, map, res, xmin, xmax, ymin, ymax, N):
    """
    Particle filter for localization
    """
    weights = update_step(lidar, particles, weights, map, res, xmin, xmax, ymin, ymax, N)

def particle_filter(lidar, v, w, particles, weights, map, res, xmin, xmax, ymin, ymax, N):
    """
    Particle filter for localization
    """
    for i in range(v.shape[0]):

        weights = update_step(lidar, particles, weights, map, res, xmin, xmax, ymin, ymax, N)
        particles = predict_step(lidar, v, w, particles, weights, map, res, xmin, xmax, ymin, ymax, N)

    return particles, weights

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
        t = np.array([[x], [y - 0.3]])

        lidar_coord = np.vstack((x_lidar, y_lidar))
        map_coord = (R @ lidar_coord) + t

        map_coord[0,:] = np.ceil((map_coord[0,:] - xmin) / res ).astype(np.int16)-1
        map_coord[1,:] = np.ceil((map_coord[1,:] - ymin) / res ).astype(np.int16)-1

        x_map = np.ceil((x - xmin) / res ).astype(np.int16)-1
        y_map = np.ceil((y - ymin) / res ).astype(np.int16)-1

        for l in range(map_coord.shape[1]):
            x_occ, y_occ = bresenham2D(x_map, y_map, map_coord[0, l], map_coord[1, l])
            for m in range(x_occ.shape[0]):
                if m == 0 or m == x_occ.shape[0] - 1:
                    map[int(x_occ[m]), int(y_occ[m])] = 0
                else:
                    map[int(x_occ[m]), int(y_occ[m])] = 1
        
        pose[0] += v[i] * np.cos(pose[2]) * 0.025
        pose[1] += v[i] * np.sin(pose[2]) * 0.025
        pose[2] += w[i] * 0.025
        
        print("Dead Reckoning: ", pose)
        print(i)
        print("Time: ", time.time() - start)
        print("=================================")
    
    return map