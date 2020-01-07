import numpy as np
import cv2 # only for valid_area


class OcamCamera:
    """ using Ocamcalib """

    def __init__(self, filename, show_flag=True):
        with open(filename, "r") as file:
            lines = file.readlines()
        calibdata = []
        for line in lines:
            if (line[0] == '#' or line[0] == '\n'):
                continue
            calibdata.append([float(i) for i in line.split()])

        # polynomial coefficients for the DIRECT mapping function
        self.pol = calibdata[0][1:]
        # polynomial coefficients for the inverse mapping function
        self.invpol = calibdata[1][1:]
        # center: "row" and "column", starting from 0 (C convention)
        self.xc = calibdata[2][0]
        self.yc = calibdata[2][1]
        # affine parameters "c", "d", "e"
        self.affine = calibdata[3]
        # image size: "height" and "width"
        self.img_size = (int(calibdata[4][0]), int(calibdata[4][1]))

        if (show_flag):
            print("pol :" + str(self.pol))
            print("invpol :" + str(self.invpol))
            print("xc :" + str(self.xc) + "\tyc :" + str(self.yc))
            print("affine :" + str(self.affine))
            print("img_size :" + str(self.img_size))

    def cam2world(self, point2D):
        """
        CAM2WORLD projects a 2D point onto the unit sphere
        The coordinate is different than that of the original ocamcalib
        Point3D coord: x:right direction, y:down direction, z:front direction
        Point2D coord: x:row direction, y:col direction (OpenCV image coordinate)
        :param np.array point2D: array of point in image 2xN
        :return np.array point3D: array of point on unit sphere 3xN
        """
        # in case of point2D = list([u, v])
        if isinstance(point2D, list):
            point2D = np.array(point2D)
        if point2D.ndim == 1:
            point2D = point2D[:, np.newaxis]
        assert point2D.shape[0] == 2

        invdet = 1 / (self.affine[0] - self.affine[1] * self.affine[2])
        xp = invdet * ((point2D[1] - self.xc) - self.affine[1] * (point2D[0] - self.yc))
        yp = invdet * (-self.affine[2] * (point2D[1] - self.xc) + self.affine[0] * (point2D[0] - self.yc))

        # distance [pixels] of  the point from the image center
        r = np.sqrt(xp * xp + yp * yp)
        # be careful about z axis direction
        zp = -np.array([elment * r ** i for (i, elment) in enumerate(self.pol)]).sum(axis=0)
        # normalize to unit norm
        point3D = np.stack([yp, xp, zp])
        point3D /= np.linalg.norm(point3D, axis=0)
        return point3D

    def world2cam(self, point3D):
        """
        WORLD2CAM projects a 3D point on to the image
        The coordinate is different than that of the original ocamcalib
        Point3D coord: x:right direction, y:down direction, z:front direction
        Point2D coord: x:row direction, y:col direction (OpenCV image coordinate)
        :param np.array point3D: array of point in cam coord 3xN
        :return np.array point2D: array of point in image 2xN
        """
        # in case of point3D = list([x,y,z])
        if isinstance(point3D, list):
            point3D = np.array(point3D)
        if point3D.ndim == 1:
            point3D = point3D[:, np.newaxis]
        assert point3D.shape[0] == 3

        # return value
        point2D = np.zeros((2, point3D.shape[1]), dtype=np.float32)

        norm = np.sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1])
        valid_flag = (norm != 0)

        # optical center
        point2D[0][~valid_flag] = self.yc
        point2D[1][~valid_flag] = self.xc

        # else
        theta = -np.arctan(point3D[2][valid_flag] / norm[valid_flag])
        invnorm = 1 / norm[valid_flag]
        rho = np.array([elment * theta ** i for (i, elment) in enumerate(self.invpol)]).sum(axis=0)
        u = point3D[0][valid_flag] * invnorm * rho
        v = point3D[1][valid_flag] * invnorm * rho
        point2D[0][valid_flag] = v * self.affine[2] + u + self.yc
        point2D[1][valid_flag] = v * self.affine[0] + u * self.affine[1] + self.xc
        return point2D

    def valid_area(self, fov=180):
        """
        get valid area based on fov. skew parameter is not considered for simplicity
        :param float : fov in degree
        :return np.array : mask 255:inside fov, 0:outside fov
        """
        valid = np.zeros(self.img_size, dtype=np.uint8)
        theta = np.deg2rad(fov / 2) - np.pi / 2
        rho = sum([elment * theta ** i for (i, elment) in enumerate(self.invpol)])
        cv2.ellipse(valid, ((self.yc, self.xc), (2 * rho, 2 * rho * self.affine[0]), 0), (255), -1)
        return valid


if __name__ == '__main__':
    ocam_file = './calib_results_0.txt'
    ocam = OcamCamera(ocam_file)
    error = []
    for _ in range(100):
        point2D = np.random.rand(2) * ocam.xc
        point3D = ocam.cam2world(point2D)
        reproj = ocam.world2cam(point3D)
        error.append(np.linalg.norm(point2D - reproj.flatten()))
        if error[-1] > 0.05:
            print("error is too big")
            print(point2D, reproj.flatten())

    print(f'Max reprojection error is {max(error):.4}.')
