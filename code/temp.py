import numpy as np

br = np.zeros((1,2))
"""
    for i in range(x_lidar[0].shape[0]):
        temp = (bresenham2D(0, 0, x_lidar[0][i], y_lidar[0][i]))
        br = np.concatenate((br, temp.T), axis=0)
    #map = np.ones((20, 20))
    br = br[1:]
    for i in range(br.shape[0]):
        map_2[34 - int(br[i][1]), 34 - int(br[i][0])] = 0
    
    #plt.imshow(map_2, cmap="gray")
    #plt.savefig("map.png")
    #plt.scatter(x_lidar[::50], y_lidar[::50], s=0.1)
    #plt.scatter(34 - x_lidar[0], 34 - y_lidar[0], color="red", s=0.1)
"""