import numpy as np
import cv2  # only for valid_area


class OcamCamera:
    """ using Ocamcalib """

    def __init__(self, filename, show_flag=False):
        with open(filename, "r") as file:
            lines = file.readlines()
        calibdata = []
        for line in lines:
            if (line[0] == '#' or line[0] == '\n'):
                continue
            calibdata.append([float(i) for i in line.split()])

        # polynomial coefficients for the DIRECT mapping function
        self._pol = calibdata[0][1:]
        # polynomial coefficients for the inverse mapping function
        self._invpol = calibdata[1][1:]
        # center: "row" and "column", starting from 0 (C convention)
        self._xc = calibdata[2][0]
        self._yc = calibdata[2][1]
        # _affine parameters "c", "d", "e"
        self._affine = calibdata[3]
        # image size: "height" and "width"
        self._img_size = (int(calibdata[4][0]), int(calibdata[4][1]))

        if show_flag:
            print(self)

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

        invdet = 1 / (self._affine[0] - self._affine[1] * self._affine[2])
        xp = invdet * ((point2D[1] - self._xc) - self._affine[1] * (point2D[0] - self._yc))
        yp = invdet * (-self._affine[2] * (point2D[1] - self._xc) + self._affine[0] * (point2D[0] - self._yc))

        # distance [pixels] of  the point from the image center
        r = np.sqrt(xp * xp + yp * yp)
        # be careful about z axis direction
        zp = -np.array([elment * r ** i for (i, elment) in enumerate(self._pol)]).sum(axis=0)
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
        point2D[0][~valid_flag] = self._yc
        point2D[1][~valid_flag] = self._xc

        # else
        theta = -np.arctan(point3D[2][valid_flag] / norm[valid_flag])
        invnorm = 1 / norm[valid_flag]
        rho = np.array([elment * theta ** i for (i, elment) in enumerate(self._invpol)]).sum(axis=0)
        u = point3D[0][valid_flag] * invnorm * rho
        v = point3D[1][valid_flag] * invnorm * rho
        point2D[0][valid_flag] = v * self._affine[2] + u + self._yc
        point2D[1][valid_flag] = v * self._affine[0] + u * self._affine[1] + self._xc
        return point2D

    def valid_area(self, fov=180):
        """
        get valid area based on fov. skew parameter is not considered for simplicity
        :param float : fov in degree
        :return np.array : mask 255:inside fov, 0:outside fov
        """
        valid = np.zeros(self._img_size, dtype=np.uint8)
        theta = np.deg2rad(fov / 2) - np.pi / 2
        rho = sum([elment * theta ** i for (i, elment) in enumerate(self._invpol)])
        cv2.ellipse(valid, ((self._yc, self._xc), (2 * rho, 2 * rho * self._affine[0]), 0), (255), -1)
        return valid

    @property
    def width(self):
        return self._img_size[1]

    @property
    def height(self):
        return self._img_size[0]

    def __repr__(self):
        print_list = []
        print_list.append(f"pol: {self._pol}")
        print_list.append(f"invpol: {self._invpol}")
        print_list.append(f"xc(col dir): {self._xc}, \tyc(row dir): {self._yc} in Ocam coord")
        print_list.append(f"affine: {self._affine}")
        print_list.append(f"img_size: {self._img_size}")
        return "\n".join(print_list)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()


if __name__ == '__main__':
    ocam_file = './calib_results_0.txt'
    ocam = OcamCamera(ocam_file)
    error = []
    for _ in range(100):
        point2D = np.random.rand(2) * ocam._xc
        point3D = ocam.cam2world(point2D)
        reproj = ocam.world2cam(point3D)
        error.append(np.linalg.norm(point2D - reproj.flatten()))
        if error[-1] > 0.05:
            print("error is too big")
            print(point2D, reproj.flatten())

    print(f'Max reprojection error is {max(error):.4}.')
