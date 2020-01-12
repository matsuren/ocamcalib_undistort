import cv2
import numpy as np

from ocamcamera import OcamCamera


def main():
    ocam_file = '../ocamcamera/calib_results_0.txt'
    img = cv2.imread('../ocamcamera/img0.jpg')
    ocam = OcamCamera(ocam_file, fov=185)
    print(ocam)

    # valid area
    valid = ocam.valid_area()

    # Perspective projection
    W = 500
    H = 500
    z = W / 3.0
    x = [i - W / 2 for i in range(W)]
    y = [j - H / 2 for j in range(H)]
    x_grid, y_grid = np.meshgrid(x, y, sparse=False, indexing='xy')
    point3D = np.stack([x_grid, y_grid, np.full_like(x_grid, z)]).reshape(3, -1)
    mapx, mapy = ocam.world2cam(point3D)
    mapx = mapx.reshape(H, W)
    mapy = mapy.reshape(H, W)
    pers_out = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # Equirectangular projection
    W = 800
    H = 400
    th = np.pi / H
    p = 2 * np.pi / W
    phi = [-np.pi + (i + 0.5) * p for i in range(W)]
    theta = [-np.pi / 2 + (i + 0.5) * th for i in range(H)]
    phi_xy, theta_xy = np.meshgrid(phi, theta, sparse=False, indexing='xy')
    point3D = np.stack(
        [np.sin(phi_xy) * np.cos(theta_xy), np.sin(theta_xy), np.cos(phi_xy) * np.cos(theta_xy)]).reshape(3, -1)
    mapx, mapy = ocam.world2cam(point3D)
    mapx = mapx.reshape(H, W)
    mapy = mapy.reshape(H, W)
    equi_out = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # Visualize images
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    valid = cv2.resize(valid, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("src", img)
    cv2.imshow("valid", valid)
    cv2.imshow("Perspecive projection", pers_out)
    cv2.imshow("Equirectangular projection", equi_out)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
