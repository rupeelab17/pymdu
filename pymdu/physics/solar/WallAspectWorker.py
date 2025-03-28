import linecache
import math
import sys

import numpy as np

from pymdu.physics.solar.WallAlgoritms import get_ders

try:
    import scipy.ndimage as sc
except:
    pass


class WallAspectWorker(object):
    def __init__(
        self,
        walls,
        scale,
        dsm,
    ):
        self.killed = False
        self.dsm = dsm
        self.scale = scale
        self.walls = walls

    def run(self):
        ret = None
        try:
            a = self.dsm
            scale = self.scale
            walls = self.walls
            # def filter1Goodwin_as_aspect_v3(walls, scale, a):

            row = a.shape[0]
            col = a.shape[1]

            filtersize = np.floor((scale + 0.0000000001) * 9)
            if filtersize <= 2:
                filtersize = 3
            else:
                if filtersize != 9:
                    if filtersize % 2 == 0:
                        filtersize = filtersize + 1

            filthalveceil = int(np.ceil(filtersize / 2.0))
            filthalvefloor = int(np.floor(filtersize / 2.0))

            filtmatrix = np.zeros((int(filtersize), int(filtersize)))
            buildfilt = np.zeros((int(filtersize), int(filtersize)))

            filtmatrix[:, filthalveceil - 1] = 1
            buildfilt[filthalveceil - 1, 0:filthalvefloor] = 1
            buildfilt[filthalveceil - 1, filthalveceil : int(filtersize)] = 2

            y = np.zeros((row, col))  # final direction
            z = np.zeros((row, col))  # temporary direction
            x = np.zeros((row, col))  # building side
            walls[walls > 0] = 1

            for h in range(0, 180):  # =0:1:180 #%increased resolution to 1 deg 20140911
                if self.killed is True:
                    break
                # print h
                # filtmatrix1temp = sc.imrotate(filtmatrix, h, 'bilinear')
                # filtmatrix1 = np.round(filtmatrix1temp / 255.)
                # filtmatrixbuildtemp = sc.imrotate(buildfilt, h, 'nearest')
                # filtmatrixbuild = np.round(filtmatrixbuildtemp / 127.)
                filtmatrix1temp = sc.rotate(
                    filtmatrix, h, order=1, reshape=False, mode='nearest'
                )  # bilinear
                filtmatrix1 = np.round(filtmatrix1temp)
                filtmatrixbuildtemp = sc.rotate(
                    buildfilt, h, order=0, reshape=False, mode='nearest'
                )  # Nearest neighbor
                filtmatrixbuild = np.round(filtmatrixbuildtemp)
                index = 270 - h
                if h == 150:
                    filtmatrixbuild[:, filtmatrix.shape[0] - 1] = 0
                if h == 30:
                    filtmatrixbuild[:, filtmatrix.shape[0] - 1] = 0
                if index == 225:
                    n = filtmatrix.shape[0] - 1
                    filtmatrix1[0, 0] = 1
                    filtmatrix1[n, n] = 1
                if index == 135:
                    n = filtmatrix.shape[0] - 1
                    filtmatrix1[0, n] = 1
                    filtmatrix1[n, 0] = 1

                for i in range(
                    int(filthalveceil) - 1, row - int(filthalveceil) - 1
                ):  # i=filthalveceil:sizey-filthalveceil
                    for j in range(
                        int(filthalveceil) - 1, col - int(filthalveceil) - 1
                    ):  # (j=filthalveceil:sizex-filthalveceil
                        if walls[i, j] == 1:
                            wallscut = (
                                walls[
                                    i - filthalvefloor : i + filthalvefloor + 1,
                                    j - filthalvefloor : j + filthalvefloor + 1,
                                ]
                                * filtmatrix1
                            )
                            dsmcut = a[
                                i - filthalvefloor : i + filthalvefloor + 1,
                                j - filthalvefloor : j + filthalvefloor + 1,
                            ]
                            if z[i, j] < wallscut.sum():  # sum(sum(wallscut))
                                z[i, j] = wallscut.sum()  # sum(sum(wallscut));
                                if np.sum(dsmcut[filtmatrixbuild == 1]) > np.sum(
                                    dsmcut[filtmatrixbuild == 2]
                                ):
                                    x[i, j] = 1
                                else:
                                    x[i, j] = 2

                                y[i, j] = index

            y[(x == 1)] = y[(x == 1)] - 180
            y[(y < 0)] = y[(y < 0)] + 360

            grad, asp = get_ders(a, scale)

            y = y + ((walls == 1) * 1) * ((y == 0) * 1) * (asp / (math.pi / 180.0))

            dirwalls = y

            wallresult = {'dirwalls': dirwalls}

            # return dirwalls

            # for i in range(skyvaultaltint.size):
            #     for j in range(aziinterval[i]):
            #
            #         if self.killed is True:
            #             break

            if self.killed is False:
                ret = wallresult
        except Exception:
            errorstring = self.print_exception()
            print(errorstring)

        return ret

    def print_exception(self):
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        return 'EXCEPTION IN {}, \nLINE {} "{}" \nERROR MESSAGE: {}'.format(
            filename, lineno, line.strip(), exc_obj
        )
