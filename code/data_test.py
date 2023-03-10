import numpy as np
import matplotlib.pyplot as plt
import math
from pr2_utils import bresenham2D


def lp_filter(arr, alpha = 0.1):
    fil_arr = [arr[0]]
    for a in arr[1:]:
        fil_arr.append(alpha*a+(1-alpha)*fil_arr[-1])
    return np.array(fil_arr)

def synchronize_lidar(lidar, v, w, ts):
    
    i = 0
    j = 0
    ranges = []
    v_temp = []
    w_temp = []
    ts_new = []

    while (i < ts.shape[0]) and (j < lidar["range"].shape[1]):
        if abs(lidar["timestamps"][j] - ts[i]) < 0.2:
            ranges.append(lidar["range"][:, j])
            v_temp.append(v[i])
            w_temp.append(w[i])
            ts_new.append(ts[i])
            i += 1
            j += 1
        elif lidar["timestamps"][j] > ts[i]:
            i += 1
        elif lidar["timestamps"][j] < ts[i]:
            j += 1
    
    timestep = 0
    for i in range(len(ts_new)):
        if i == 0:
            continue
        timestep += ts_new[i] - ts_new[i-1]
    
    timestep = timestep / len(ts_new)
    print("Timestep: ", timestep)  
    ranges = np.array(ranges)
    v_temp = np.array(v_temp)
    w_temp = np.array(w_temp)
    ts_new = np.array(ts_new)
    return ranges, v_temp, w_temp, ts_new

def synchronize_data(encoder, imu, v_enc, w_imu):

    v = []
    w = []
    ts = []
    j = 0
    w_imu = lp_filter(w_imu)

    for i in range(w_imu.shape[0]):
        if abs(imu["timestamps"][i] - encoder["timestamps"][j]) <= 0.01:
            w.append(w_imu[i])
            v.append(v_enc[j])
            ts.append(imu["timestamps"][i])
            j += 1

        if encoder["timestamps"][j] < imu["timestamps"][i]:
            j += 1
    
    w = np.array(w)
    v = np.array(v)
    ts = np.array(ts)

    return v, w, ts

def motion_model(v, w, time):
    pose = [0, 0, 0]
    states = []
    states.append(pose)
    x = []
    y = []
    theta = []
    for i in range(v.shape[0]):
        pose[0] += v[i] * np.cos(pose[2]) * time
        pose[1] += v[i] * np.sin(pose[2]) * time
        pose[2] += w[i] * time
        x.append(pose[0])
        y.append(pose[1])
        theta.append(pose[2])
        states.append(pose)
    x=np.array(x)
    y=np.array(y)
    theta=np.array(theta)
    states = np.array(states)

    return x, y, theta, states


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