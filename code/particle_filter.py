import numpy as np
import matplotlib.pyplot as plt
from load_data import load
from data_test import *
import math
from pr2_utils import *


def first_scan(lidar, map, res, xmin, xmax, ymin, ymax):
    """
    First scan of the particle filter
    """
    scan = lidar[:,0]
    ranges = scan
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    theta = 0
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles) + 0.3
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    lidar_coord = np.array([x, y])
    lidar_coord = R @ lidar_coord

    x_ranges = np.ceil((lidar_coord[0,:] - xmin) / res ).astype(np.int16)-1
    y_ranges = np.ceil((lidar_coord[1,:] - ymin) / res ).astype(np.int16)-1

    x_map = np.ceil((0 - xmin) / res ).astype(np.int16)-1
    y_map = np.ceil((0 - ymin) / res ).astype(np.int16)-1

    x_obs = x_ranges
    y_obs = y_ranges
    
    for l in range(x_obs.shape[0]):
        x_occ, y_occ = bresenham2D(x_map, y_map, x_obs[l], y_obs[l])
        for m in range(x_occ.shape[0]):
            if (m == x_occ.shape[0]-1): 
                map[int(x_occ[m])][int(y_occ[m])] -= np.log(4)
            else:
                map[int(x_occ[m])][int(y_occ[m])] += np.log(4)
            if map[int(x_occ[m])][int(y_occ[m])] < -(4 * np.log(4)):
                map[int(x_occ[m])][int(y_occ[m])] = -(4 * np.log(4))
            if map[int(x_occ[m])][int(y_occ[m])] > (4 * np.log(4)):
                map[int(x_occ[m])][int(y_occ[m])] = (4 * np.log(4))          
    
    return map

def update(t, lidar, particles, weights, map, res, xmin, xmax, ymin, ymax, N):

    dtheta = 0.6
    theta_cap = np.arange(-dtheta, dtheta, dtheta / 6)
    R_cap = np.array([[np.cos(theta_cap), -np.sin(theta_cap)], [np.sin(theta_cap), np.cos(theta_cap)]])
    R_cap = np.transpose(R_cap, (2, 0, 1))
    scan = lidar[:,t]
    ranges = scan
    angles = np.arange(-135,135.25,0.25)*np.pi/180.0
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]

    x_lidar = ranges * np.cos(angles)
    y_lidar = ranges * np.sin(angles) + 0.3

    lidar_coord = np.array([x_lidar, y_lidar])

    theta = particles[: , 2]
    
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    R = np.transpose(R, (2, 0, 1))
    t = np.array([[particles[:, 0]], [particles[:, 1]]])
    t = np.transpose(t, (2, 0, 1))

    map_coord_N = R @ lidar_coord
    corr = np.zeros(N)
    theta_change = np.zeros(N)

    for j in range(N):
        temp_map = R_cap @ map_coord_N[j, :, :]
        temp_corr = np.zeros(theta_cap.shape[0])
        for k in range(theta_cap.shape[0]):
            temp_map[k, :, :] = np.ceil(((temp_map[k, :, :] + t[j]) - np.array([[xmin], [ymin]])) / res ).astype(np.int16)-1
            for i in range(temp_map.shape[2]):
                if map[int(temp_map[k, 0, i]), int(temp_map[k, 1, i])] < 0:
                    temp_corr[k] += abs(map[int(temp_map[k, 0, i]), int(temp_map[k, 1, i])])
        
        theta_change[j] = np.argmax(temp_corr)
        corr[j] = np.max(temp_corr)

    weights = weights * corr
    weights = weights / np.sum(weights)
        
    update_id = np.argmax(weights)
    
    map_coord = map_coord_N[update_id, :, :]

    map_coord = R_cap[int(theta_change[update_id]), :, :] @ map_coord

    map_coord = np.ceil(((map_coord + t[update_id]) - np.array([[xmin], [ymin]])) / res ).astype(np.int16)-1

    x = particles[update_id, 0]
    y = particles[update_id, 1]

    x = np.ceil((x - xmin) / res ).astype(np.int16)-1
    y = np.ceil((y - ymin) / res ).astype(np.int16)-1

    for l in range(map_coord.shape[1]):
        x_occ, y_occ = bresenham2D(x, y, map_coord[0, l], map_coord[1, l])
        for m in range(x_occ.shape[0]):
            if m == (x_occ.shape[0] - 1):
                map[int(x_occ[m]), int(y_occ[m])] -= math.log(4)
            else:
                map[int(x_occ[m]), int(y_occ[m])] += math.log(4)
            if map[int(x_occ[m]), int(y_occ[m])] < -(4 * math.log(4)):
                map[int(x_occ[m]), int(y_occ[m])] = -(4 * math.log(4))
            if map[int(x_occ[m]), int(y_occ[m])] > (4 * math.log(4)):
                map[int(x_occ[m]), int(y_occ[m])] = (4 * math.log(4))

    return weights, map


def predict_step(v, w, particles):
    """
    Particle filter for localization
    """
    noise_v = np.random.normal(0, (0.2 * abs(v)), (particles.shape[0], 1))
    noise_w = np.random.normal(0, (0.4 * abs(w)), (particles.shape[0], 1))
    angles = particles[:, 2]
    particles[:,0] = particles[:,0] + ((v + noise_v[:,0]) * np.cos(angles)) * 0.025
    particles[:,1] = particles[:,1] + ((v + noise_v[:,0]) * np.sin(angles)) * 0.025
    particles[:,2] = particles[:,2] + (w + noise_w[:,0]) * 0.025
        
    return particles

def resample(weights, particles, N):
    """
    Resample particles
    """
    weights_temp = weights
    particles_temp = particles
    for i in range(N):
        weights = weights / np.sum(weights)
        if weights[i] <= (0.25 * (1 / N)):
            index = np.random.choice(N, 1, p=weights)
            weights_temp[i] = 1 / N
            particles_temp[i] = particles[index]
    weights = weights_temp
    particles = particles_temp
    weights = weights / np.sum(weights)
    return particles, weights

def particle_filter(lidar, v, w, particles, weights, map, res, xmin, xmax, ymin, ymax, N):
    """
    Particle filter for localization
    """
    for i in range(lidar.shape[1]):

        start = time.time()

        particles = predict_step(v[i], w[i], particles)
        weights, map = update(i, lidar, particles, weights, map, res, xmin, xmax, ymin, ymax, N)
        print("Weights: ", weights)
        print(i)
        print("Time: ", time.time() - start)
        print("=================================")
        particles, weights = resample(weights, particles, N)

        if i % 800 == 0:
            fig = plt.figure()
            plt.imshow(map, cmap='gray')
            x_temp = particles[:, 0]
            y_temp = particles[:, 1]
            x_temp = np.ceil((x_temp[:] - xmin) / res ).astype(np.int16)-1
            y_temp = np.ceil((y_temp[:] - ymin) / res ).astype(np.int16)-1
            print(x_temp)
            print(y_temp)
            plt.scatter(y_temp, x_temp, c='r', s=1)
            plt.savefig('map_' + str(i) + '.png')
            
    return particles, weights, map

def dead_reckon (lidar, v, w, map, res, xmin, xmax, ymin, ymax):
    """
    Dead reckoning for localization
    """
    pose = [0, 0, 0]
    for i in range(lidar.shape[1]):
        pose[0] += (v[i] * np.cos(pose[2]) * 0.025)
        pose[1] += (v[i] * np.sin(pose[2]) * 0.025)
        pose[2] += (w[i] * 0.025)
        start = time.time()
        scan = lidar[: , i]
        ranges = scan
        angles = np.arange(-135,135.25,0.25)*np.pi/180.0
        indValid = np.logical_and((ranges < 30),(ranges> 0.1))
        ranges = ranges[indValid]
        angles = angles[indValid]
        x_lidar = ranges * np.cos(angles)
        y_lidar = ranges * np.sin(angles) + 0.3

        x = pose[0]
        y = pose[1]

        theta = pose[2]
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        t = np.array([[x], [y]])

        lidar_coord = np.array([x_lidar, y_lidar])
        lidar_coord = (R @ lidar_coord) + t

        map_coord = np.ceil((lidar_coord[:,:] - np.array([[xmin], [ymin]])) / res ).astype(np.int16)-1

        x_map = np.ceil((x - xmin) / res ).astype(np.int16)-1
        y_map = np.ceil((y - ymin) / res ).astype(np.int16)-1

        x_obs = map_coord[0,:]
        y_obs = map_coord[1,:]

        for l in range(x_obs.shape[0]):
            x_occ, y_occ = bresenham2D(x_map, y_map, x_obs[l], y_obs[l])
            for m in range(x_occ.shape[0]):
                if (m == x_occ.shape[0]-1): 
                    map[int(x_occ[m])][int(y_occ[m])] -= np.log(4)
                else: 
                    map[int(x_occ[m])][int(y_occ[m])] += np.log(4)
                if map[int(x_occ[m])][int(y_occ[m])] < -4:
                    map[int(x_occ[m])][int(y_occ[m])] = -4
                if map[int(x_occ[m])][int(y_occ[m])] > 4:
                    map[int(x_occ[m])][int(y_occ[m])] = 4
        
        print("Dead Reckoning: ", x_map, y_map, pose[2])
        print(i)
        print("Time: ", time.time() - start)
        print("=================================")
    
    return map