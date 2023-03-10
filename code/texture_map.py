import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

disp_path = "./../../data/dataRGBD/Disparity20/"
rgb_path = "./../../data/dataRGBD/RGB20/"

def normalize(img):
    max_ = img.max()
    min_ = img.min()
    return (img - min_)/(max_-min_)

def texture_map_plot(dataset, ts_kinect, ts_depth, map, res, xmin, xmax, ymin, ymax, x, y, theta, path_rgb, path_depth, ts_occ):

    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713

    R = np.array([[0.9356, 0.0209, -0.3522], [-0.0197, 0.9997, 0.0073], [0.3522, 0, 0.9358]])
    p = np.array([0.33276, 0, 0.38001])

    texture = np.zeros((map.shape[0], map.shape[1], 3))
    texture_array = []
    
    for i in range(ts_kinect.shape[0]):

        if i == 0:
            continue

        start_time = time.time()

        idx = np.argmin(np.abs(ts_kinect[i] - ts_occ))
        depth_id = np.argmin(np.abs(ts_kinect[i] - ts_depth))
        rgb = cv2.imread(path_rgb + "rgb" + dataset + "_" + str(i) + ".png")[...,::-1]
        depth = cv2.imread(path_depth + "disparity" + dataset + "_" + str(depth_id) + ".png", cv2.IMREAD_UNCHANGED)

        rgbi, rgbj, valid, depth = read_disparity_img(depth)

        temp_1 = (rgbi * depth).flatten()
        temp_2 = (rgbj * depth).flatten()
        temp_3 = (depth.flatten())
        temp = np.vstack((temp_1, temp_2, temp_3))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsic = np.linalg.inv(K)
        temp = intrinsic @ temp
        Ror = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        Ror = Ror.T
        temp = Ror @ temp
        temp = R @ temp + p.reshape(3, 1)
        
        R_pose = np.array([[np.cos(theta[idx]), -np.sin(theta[idx]), 0], [np.sin(theta[idx]), np.cos(theta[idx]), 0], [0, 0, 1]])
        t = np.array([x[idx], y[idx], 0.127])
        temp = R_pose @ temp + t.reshape(3, 1)
        temp_xy = temp[:2, :]
        #temp_xy = temp_xy / temp[2, :]

        temp_xy = np.ceil((temp_xy[:,:] - np.array([[xmin], [ymin]])) / res).astype(np.int16)-1
        temp_x = temp_xy[0, :]
        temp_y = temp_xy[1, :]
        temp_x = temp_x.reshape((480, 640))
        temp_y = temp_y.reshape((480, 640))

        for j in range(480):
            for k in range(640):
                if valid[j, k]:
                    texture[temp_x[j, k], temp_y[j, k], :] = (rgb[j, k, :])
        
        print(i)
        print("Texture map", i, "out of", ts_kinect.shape[0], "done.")
        print("Time taken:", time.time() - start_time, "seconds")
        print("Estimated time remaining:", (time.time() - start_time) * (ts_kinect.shape[0] - i), "seconds")
        print("============================================================================================")

        if i % 500 == 0:
            fig = plt.figure()
            plt.imshow(texture.astype(np.uint8))
            plt.savefig("texture_" + str(i) + ".png")        
    
    return texture


def read_disparity_img(imd):
    disparity = imd.astype(np.float32)

    # get depth
    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd

    # calculate u and v coordinates 
    v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
    #u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))

    # get 3D coordinates 
    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713
    x = (u-cx) / fx * z
    y = (v-cy) / fy * z

    # calculate the location of each pixel in the RGB image
    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
    rgbv = np.round((v * 526.37 + 16662.0)/fy)
    valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])

    return rgbu, rgbv, valid, z


if __name__ == '__main__':

    # load RGBD image
    imd = cv2.imread(disp_path+'disparity20_2.png',cv2.IMREAD_UNCHANGED) # (480 x 640)
    imc = cv2.imread(rgb_path+'rgb20_2.png')[...,::-1] # (480 x 640 x 3)

    #print(imc.shape)

    # convert from disparity from uint16 to double
    disparity = imd.astype(np.float32)

    # get depth
    dd = (-0.00304 * disparity + 3.31)
    z = 1.03 / dd

    # calculate u and v coordinates 
    v,u = np.mgrid[0:disparity.shape[0],0:disparity.shape[1]]
    #u,v = np.meshgrid(np.arange(disparity.shape[1]),np.arange(disparity.shape[0]))

    # get 3D coordinates 
    fx = 585.05108211
    fy = 585.05108211
    cx = 315.83800193
    cy = 242.94140713
    x = (u-cx) / fx * z
    y = (v-cy) / fy * z

    # calculate the location of each pixel in the RGB image
    rgbu = np.round((u * 526.37 + dd*(-4.5*1750.46) + 19276.0)/fx)
    rgbv = np.round((v * 526.37 + 16662.0)/fy)
    valid = (rgbu>= 0)&(rgbu < disparity.shape[1])&(rgbv>=0)&(rgbv<disparity.shape[0])



    # display valid RGB pixels
    fig = plt.figure(figsize=(10, 13.3))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(z[valid],-x[valid],-y[valid],c=imc[rgbv[valid].astype(int),rgbu[valid].astype(int)]/255.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=0, azim=180)
    plt.show()

    # display disparity image
    plt.imshow(normalize(imd), cmap='gray')
    plt.show()
