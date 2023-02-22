import numpy as np
import sys
from load_data import load

if __name__ == "__main__":
    dataset = sys.argv[1]
    encoder, lidar, imu, kinect = load(int(dataset))

    enc_data = encoder["counts"]
    enc_ts = encoder["timestamps"]

    print(enc_data)

    

