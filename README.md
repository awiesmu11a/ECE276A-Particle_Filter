# ECE276A-Particle_Filter
Implementation of particle filter using sensors kinect camera, LIDAR, IMU and encoders

* Run `pip install requirements.txt ` to install required libraries.
* Run `python3 main.py [dataset id] ` to run file which generates results.
* data_test.py consist of code to preprocess the sensor, like time synchronization, low pass filter, generating pose only from IMU, generating pose only from encoders.
* particle_filter.py consist of implementation of particle filter.
* texture_map.py contains implementation of texture map part of project.
