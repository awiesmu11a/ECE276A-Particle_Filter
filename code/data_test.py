import numpy as np
import matplotlib.pyplot as plt
import math
from pr2_utils import bresenham2D

def synchronize_data(encoder, imu, v_enc, w_imu):

    v = []
    w = []
    j = 0

    for i in range(w_imu.shape[0]):
        if abs(imu["timestamps"][i] - encoder["timestamps"][j]) <= 0.01:
            w.append(w_imu[i])
            v.append(v_enc[j])
            j += 1

        if encoder["timestamps"][j] < imu["timestamps"][i]:
            j += 1
    
    w = np.array(w)
    v = np.array(v)

    return v, w

def motion_model(v, w):
    pose = [0, 0, 0]
    states = []
    states.append(pose)
    x = []
    y = []
    theta = []
    for i in range(v.shape[0]):
        pose[0] += v[i] * np.cos(pose[2]) * 0.025
        pose[1] += v[i] * np.sin(pose[2]) * 0.025
        pose[2] += w[i] * 0.025
        x.append(pose[0])
        y.append(pose[1])
        theta.append(pose[2])
        states.append(pose)
    x=np.array(x)
    y=np.array(y)
    theta=np.array(theta)
    states = np.array(states)

    return x, y, theta, states

def lidar_occupancy(lidar, pose, map):
    lidar_theta = np.zeros(lidar["range"].shape[0])
    lidar_theta[0] = lidar["angle_min"]
    for i in range(1, lidar["range"].shape[0]):
        lidar_theta[i] = lidar_theta[i-1] + lidar["angle_increment"][0][0]
    
    x_lidar = []
    y_lidar = []

    for i in range(pose.shape[0]):
        theta_temp = pose[i][2] + lidar_theta
        lidar_range_x = (lidar["range"][:, i] * np.cos(theta_temp))
        lidar_range_y = (lidar["range"][:, i] * np.sin(theta_temp))
        lidar_range_x = lidar_range_x[(lidar_range_x > 0.1) | (lidar_range_x < 30)]
        lidar_range_y = lidar_range_y[(lidar_range_y > 0.1) | (lidar_range_y < 30)]
        x_temp = np.array(pose[i][0] + lidar_range_x)
        y_temp = np.array(pose[i][1] + lidar_range_y)
        x_lidar.append(x_temp)
        y_lidar.append(y_temp)
    br = np.zeros((1, 2))
    x_lidar = np.array(x_lidar)
    y_lidar = np.array(y_lidar)
    x_lidar = x_lidar.reshape(x_lidar.shape[0] * x_lidar.shape[1])
    y_lidar = y_lidar.reshape(y_lidar.shape[0] * y_lidar.shape[1])
    for i in range(1081):
        br = np.concatenate((br, bresenham2D(pose[int(i / 1080)][0], pose[int(i / 1080)][1], x_lidar[i], y_lidar[i]).T), axis=0)
        print(i)
    
    br = br[1:, :]
    br = br.astype(int)
    for i in range(br.shape[0]):
        map[34 - br[i][0], 34 - br[i][1]] = 0
    return pose[:,0], pose[:,1], map  


def lidar_test_path(lidar, x_imu, y_imu, theta_imu, freq_match, reshape):
    lidar_theta = np.zeros(lidar["range"].shape[0])
    lidar_theta[0] = lidar["angle_min"]
    for i in range(1, lidar["range"].shape[0]):
        lidar_theta[i] = lidar_theta[i-1] + lidar["angle_increment"][0][0]
    
    x_lidar = []
    y_lidar = []

    for i in range(lidar["range"].shape[1]):
        if freq_match * i >= x_imu.shape[0]:
            break            
        theta_temp = theta_imu[freq_match * i] + lidar_theta
        lidar_range_x = (lidar["range"][:, i] * np.cos(theta_temp))
        lidar_range_y = (lidar["range"][:, i] * np.sin(theta_temp))
        lidar_range_x = lidar_range_x[(lidar_range_x > 0.1) | (lidar_range_x < 30)]
        lidar_range_y = lidar_range_y[(lidar_range_y > 0.1) | (lidar_range_y < 30)]
        x_temp = x_imu[freq_match * i] + lidar_range_x
        y_temp = y_imu[freq_match * i] + lidar_range_y
        x_lidar.append(x_temp)
        y_lidar.append(y_temp)
    
    x_lidar = np.array(x_lidar)
    y_lidar = np.array(y_lidar)
    
    if reshape:
        x_lidar = x_lidar.reshape(x_lidar.shape[0] * x_lidar.shape[1])
        y_lidar = y_lidar.reshape(y_lidar.shape[0] * y_lidar.shape[1])
    map = np.zeros((100, 450))
    #for i in range(x_lidar.shape[0]):
        #map[int(y_lidar[i] / 100) - 80, int(x_lidar[i] / 100)] = 1

    return x_lidar, y_lidar, map

def imu_test(imu):
    a_imu = imu["linear_acceleration"]
    a_imu *= 9.81
    w_imu = imu["angular_velocity"][2, :]

    v_imu = [0]
    v = np.array([0, 0], dtype=np.float64)
    for i in range(a_imu.shape[1] - 1):
        dt = imu["timestamps"][i+1] - imu["timestamps"][i]
        v += a_imu[:2, i] * dt
        v_imu.append(np.linalg.norm(v))
    
    v_imu = np.array(v_imu)

    pose = np.array([0, 0, 0])
    x = []
    y = []
    theta = []
    
    for i in range(v_imu.shape[0]):
        pose[0] += v_imu[i] * np.cos(pose[2])
        pose[1] += v_imu[i] * np.sin(pose[2])
        pose[2] += w_imu[i]
        x.append(pose[0])
        y.append(pose[1])
        theta.append(pose[2])
    
    x = np.array(x)
    y = np.array(y)
    theta = np.array(theta)

    return x, y, theta, v_imu, w_imu

def encoder_test(enc):
    enc_data = enc["counts"]
    enc_ts = enc["timestamps"]

    v_enc = []
    w_enc = []

    l = 0.3937

    for i in range(enc_data.shape[1] - 1):
        dt = enc_ts[i + 1] - enc_ts[i]
        dR = (enc_data[0, i] + enc_data[2, i]) / 2
        dL = (enc_data[1, i] + enc_data[3, i]) / 2
        v = (dR + dL) / 2
        v *= 0.0022
        v /= dt
        w = (dR - dL) / l
        w /= dt
        w *= 0.0022
        v_enc.append(v)
        w_enc.append(w)

    
    v_enc = np.array(v_enc)
    w_enc = np.array(w_enc)    

    pose = np.array([0, 0, 0])
    x = []
    y = []
    theta = []

    for i in range(v_enc.shape[0]): 
        pose[0] = pose[0] + (v_enc[i] * math.cos(pose[2]) * 0.025)
        pose[1] = pose[1] + (v_enc[i] * math.sin(pose[2]) * 0.025)
        pose[2] = pose[2] + (w_enc[i] * 0.025)
        x.append(pose[0])
        y.append(pose[1])
        theta.append(pose[2])
        
    x = np.array(x)
    y = np.array(y)
    theta = np.array(theta)

    return x, y, theta, v_enc, w_enc