# ******************************************************************************
#  This file is part of pymdu.                                                 *
#                                                                              *
#  pymdu is free software: you can redistribute it and/or modify               *
#  it under the terms of the GNU General Public License as published by        *
#  the Free Software Foundation, either version 3 of the License, or           *
#  (at your option) any later version.                                         *
#                                                                              *
#  pymdu is distributed in the hope that it will be useful,                    *
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              *
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               *
#  GNU General Public License for more details.                                *
#                                                                              *
#  You should have received a copy of the GNU General Public License           *
#  along with pymdu.  If not, see <https://www.gnu.org/licenses/>.             *
# ******************************************************************************
import linecache
import os
import sys
import zipfile

import geopandas as gpd
import numpy as np
from osgeo import gdal

from pymdu.GeoCore import GeoCore
from pymdu.physics.solar.ShadowingFunctions import (
    shadowingfunction_20,
    shadowingfunctionglobalradiation,
)
from pymdu.physics.solar.UtilitiesSolar import saveraster


class SVFCalculator(object):
    """
    ===
    Classe qui permet
    ===
    """

    def __init__(self, a, vegdem, vegdem2, scale, usevegdem):
        self.killed = False
        self.a = a
        self.vegdem = vegdem
        self.vegdem2 = vegdem2
        self.scale = scale
        self.usevegdem = usevegdem

    def run(self):
        """
        # %This m.file calculates Skyview factors on a DEM for the four cardinal points
        # %This new version is NOT using 1000 randow shadow casting, but implies
        # %the theory of annulus weights (e.g. Steyn, 1980). The number of shadow
        # %castings is reduced to 653.
        # %20130208 - changed to use cell input
        # 20181004 - New version using 145 shadow castings
        """
        ret = None
        try:
            rows = self.a.shape[0]
            cols = self.a.shape[1]
            svf = np.zeros([rows, cols])
            svfE = svf
            svfS = svf
            svfW = svf
            svfN = svf
            svfveg = np.zeros((rows, cols))
            svfEveg = np.zeros((rows, cols))
            svfSveg = np.zeros((rows, cols))
            svfWveg = np.zeros((rows, cols))
            svfNveg = np.zeros((rows, cols))
            svfaveg = np.zeros((rows, cols))
            svfEaveg = np.zeros((rows, cols))
            svfSaveg = np.zeros((rows, cols))
            svfWaveg = np.zeros((rows, cols))
            svfNaveg = np.zeros((rows, cols))

            # % amaxvalue
            vegmax = self.vegdem.max()
            amaxvalue = self.a.max()
            amaxvalue = np.maximum(amaxvalue, vegmax)

            # % Elevation vegdems if buildingDEM inclused ground heights
            self.vegdem = self.vegdem + self.a
            self.vegdem[self.vegdem == self.a] = 0
            self.vegdem2 = self.vegdem2 + self.a
            self.vegdem2[self.vegdem2 == self.a] = 0
            # % Bush separation
            bush = np.logical_not((self.vegdem2 * self.vegdem)) * self.vegdem

            # patch_option = 1 # 145 patches
            patch_option = 2  # 153 patches
            # patch_option = 3 # 306 patches
            # patch_option = 4 # 612 patches

            # Create patches based on patch_option
            # skyvaultalt, skyvaultazi, annulino, skyvaultaltint, aziinterval, skyvaultaziint, azistart= self.create_patches(
            #     patch_option)
            (
                skyvaultalt,
                skyvaultazi,
                annulino,
                skyvaultaltint,
                aziinterval,
                azistart,
            ) = self.create_patches(patch_option)

            skyvaultaziint = np.array([360 / patches for patches in aziinterval])
            iazimuth = np.hstack(np.zeros((1, np.sum(aziinterval))))  # Nils

            shmat = np.zeros((rows, cols, np.sum(aziinterval)))
            vegshmat = np.zeros((rows, cols, np.sum(aziinterval)))
            vbshvegshmat = np.zeros((rows, cols, np.sum(aziinterval)))

            index = 0

            for j in range(0, 8):
                for k in range(0, int(360 / skyvaultaziint[j])):
                    iazimuth[index] = k * skyvaultaziint[j] + azistart[j]
                    if iazimuth[index] > 360.0:
                        iazimuth[index] = iazimuth[index] - 360.0
                    index = index + 1
            aziintervalaniso = np.ceil(aziinterval / 2.0)
            index = int(0)
            # for i in np.arange(0, iangle.shape[0]-1):
            for i in range(0, skyvaultaltint.shape[0]):
                # if self.killed is True:
                #     break
                for j in np.arange(0, (aziinterval[int(i)])):
                    if self.killed is True:
                        break
                    altitude = skyvaultaltint[int(i)]
                    # azimuth = iazimuth[int(index)-1]
                    azimuth = iazimuth[int(index)]

                    # Casting shadow
                    if self.usevegdem:
                        shadowresult = shadowingfunction_20(
                            self.a,
                            self.vegdem,
                            self.vegdem2,
                            azimuth,
                            altitude,
                            self.scale,
                            amaxvalue,
                            bush,
                        )
                        vegsh = shadowresult['vegsh']
                        vbshvegsh = shadowresult['vbshvegsh']
                        sh = shadowresult['sh']
                        vegshmat[:, :, index] = vegsh
                        vbshvegshmat[:, :, index] = vbshvegsh
                    else:
                        sh = shadowingfunctionglobalradiation(
                            self.a, azimuth, altitude, self.scale, 1
                        )

                    shmat[:, :, index] = sh

                    # # Casting shadow
                    # if self.usevegdem:
                    #     shadowresult = shadow.shadowingfunction_20(self.a, self.vegdem, self.vegdem2, azimuth, altitude,
                    #                                                self.scale, amaxvalue, bush, self.dlg, 1)
                    #     vegsh = shadowresult["vegsh"]
                    #     vbshvegsh = shadowresult["vbshvegsh"]
                    #     # vegsh, sh, vbshvegsh, wallsh, wallsun, wallshve, _, facesun = shadowingfunction_wallheight_23(
                    #     #     self.a, self.vegdem, self.vegdem2, azimuth, altitude, self.scale,amaxvalue, bush,
                    #     #     self.wallheight, self.wallaspect * np.pi / 180.)
                    #     # shadow = sh - (1 - vegsh) * (1 - psi)
                    #     # wallshvemat[:, :, index] = wallshve
                    #     vegshmat[:, :, index] = vegsh
                    #     # vbshvegshmat[:, :, index] = vbshvegsh
                    # # else:
                    # #     sh, wallsh, wallsun, facesh, facesun = shadowingfunction_wallheight_13(self.a, azimuth,
                    # #                                     altitude, self.scale, self.wallheight,
                    # #                                     self.wallaspect * np.pi / 180.)
                    # sh = shadow.shadowingfunctionglobalradiation(self.a, azimuth, altitude, self.scale, self.dlg, 1)
                    #     # shadow = sh
                    # shmat[:, :, index] = sh
                    # # wallshmat[:, :, index] = wallsh
                    # # wallsunmat[:, :, index] = wallsun
                    # # facesunmat[:, :, index] = facesun
                    # # sh = shadow.shadowingfunctionglobalradiation(self.a, azimuth, altitude, self.scale, self.dlg, 1)

                    # Calculate svfs
                    for k in np.arange(
                        annulino[int(i)] + 1, (annulino[int(i + 1.0)]) + 1
                    ):
                        weight = self.annulus_weight(k, aziinterval[i]) * sh
                        svf = svf + weight
                        weight = self.annulus_weight(k, aziintervalaniso[i]) * sh
                        if (azimuth >= 0) and (azimuth < 180):
                            # weight = self.annulus_weight(k, aziintervalaniso[i])*sh
                            svfE = svfE + weight
                        if (azimuth >= 90) and (azimuth < 270):
                            # weight = self.annulus_weight(k, aziintervalaniso[i])*sh
                            svfS = svfS + weight
                        if (azimuth >= 180) and (azimuth < 360):
                            # weight = self.annulus_weight(k, aziintervalaniso[i])*sh
                            svfW = svfW + weight
                        if (azimuth >= 270) or (azimuth < 90):
                            # weight = self.annulus_weight(k, aziintervalaniso[i])*sh
                            svfN = svfN + weight

                    if self.usevegdem:
                        for k in np.arange(
                            annulino[int(i)] + 1, (annulino[int(i + 1.0)]) + 1
                        ):
                            # % changed to include 90
                            weight = self.annulus_weight(k, aziinterval[i])
                            svfveg = svfveg + weight * vegsh
                            svfaveg = svfaveg + weight * vbshvegsh
                            weight = self.annulus_weight(k, aziintervalaniso[i])
                            if (azimuth >= 0) and (azimuth < 180):
                                svfEveg = svfEveg + weight * vegsh
                                svfEaveg = svfEaveg + weight * vbshvegsh
                            if (azimuth >= 90) and (azimuth < 270):
                                svfSveg = svfSveg + weight * vegsh
                                svfSaveg = svfSaveg + weight * vbshvegsh
                            if (azimuth >= 180) and (azimuth < 360):
                                svfWveg = svfWveg + weight * vegsh
                                svfWaveg = svfWaveg + weight * vbshvegsh
                            if (azimuth >= 270) or (azimuth < 90):
                                svfNveg = svfNveg + weight * vegsh
                                svfNaveg = svfNaveg + weight * vbshvegsh

                    index += 1

            svfS = svfS + 3.0459e-004
            svfW = svfW + 3.0459e-004
            # % Last azimuth is 90. Hence, manual add of last annuli for svfS and SVFW
            # %Forcing svf not be greater than 1 (some MATLAB crazyness)
            svf[(svf > 1.0)] = 1.0
            svfE[(svfE > 1.0)] = 1.0
            svfS[(svfS > 1.0)] = 1.0
            svfW[(svfW > 1.0)] = 1.0
            svfN[(svfN > 1.0)] = 1.0

            if self.usevegdem:
                last = np.zeros((rows, cols))
                last[(self.vegdem2 == 0.0)] = 3.0459e-004
                svfSveg = svfSveg + last
                svfWveg = svfWveg + last
                svfSaveg = svfSaveg + last
                svfWaveg = svfWaveg + last
                # %Forcing svf not be greater than 1 (some MATLAB crazyness)
                svfveg[(svfveg > 1.0)] = 1.0
                svfEveg[(svfEveg > 1.0)] = 1.0
                svfSveg[(svfSveg > 1.0)] = 1.0
                svfWveg[(svfWveg > 1.0)] = 1.0
                svfNveg[(svfNveg > 1.0)] = 1.0
                svfaveg[(svfaveg > 1.0)] = 1.0
                svfEaveg[(svfEaveg > 1.0)] = 1.0
                svfSaveg[(svfSaveg > 1.0)] = 1.0
                svfWaveg[(svfWaveg > 1.0)] = 1.0
                svfNaveg[(svfNaveg > 1.0)] = 1.0

            svfresult = {
                'svf': svf,
                'svfE': svfE,
                'svfS': svfS,
                'svfW': svfW,
                'svfN': svfN,
                'svfveg': svfveg,
                'svfEveg': svfEveg,
                'svfSveg': svfSveg,
                'svfWveg': svfWveg,
                'svfNveg': svfNveg,
                'svfaveg': svfaveg,
                'svfEaveg': svfEaveg,
                'svfSaveg': svfSaveg,
                'svfWaveg': svfWaveg,
                'svfNaveg': svfNaveg,
                'shmat': shmat,
                'vegshmat': vegshmat,
                'vbshvegshmat': vbshvegshmat,
            }

            if self.killed is False:
                ret = svfresult
        except Exception as e:
            print('ERROR', e)
            # forward the exception upstream
            print(self.print_exception())

        return ret

    @staticmethod
    def print_exception():
        exc_type, exc_obj, tb = sys.exc_info()
        f = tb.tb_frame
        lineno = tb.tb_lineno
        filename = f.f_code.co_filename
        linecache.checkcache(filename)
        line = linecache.getline(filename, lineno, f.f_globals)
        return 'EXCEPTION IN {}, \nLINE {} "{}" \nERROR MESSAGE: {}'.format(
            filename, lineno, line.strip(), exc_obj
        )

    def to_gdf(self) -> gpd.GeoDataFrame:
        return

    @staticmethod
    def annulus_weight(altitude, aziinterval):
        n = 90.0
        steprad = (360.0 / aziinterval) * (np.pi / 180.0)
        annulus = 91.0 - altitude
        w = (
            (1.0 / (2.0 * np.pi))
            * np.sin(np.pi / (2.0 * n))
            * np.sin((np.pi * (2.0 * annulus - 1.0)) / (2.0 * n))
        )
        weight = steprad * w
        return weight

    @staticmethod
    def create_patches(patch_option):
        deg2rad = np.pi / 180

        # patch_option = 1 = 145 patches (Robinson & Stone, 2004)
        # patch_option = 2 = 153 patches (Wallenberg et al., 2022)
        # patch_option = 3 = 306 patches -> test
        # patch_option = 4 = 612 patches -> test

        skyvaultalt = np.atleast_2d([])
        skyvaultazi = np.atleast_2d([])

        # Creating skyvault of patches of constant radians (Tregeneza and Sharples, 1993)
        # Patch option 1, 145 patches, Original Robinson & Stone (2004) after Tregenza (1987)/Tregenza & Sharples (1993)
        if patch_option == 1:
            annulino = np.array([0, 12, 24, 36, 48, 60, 72, 84, 90])
            skyvaultaltint = np.array(
                [6, 18, 30, 42, 54, 66, 78, 90]
            )  # Robinson & Stone (2004)
            azistart = np.array([0, 4, 2, 5, 8, 0, 10, 0])  # Fredrik/Nils
            patches_in_band = np.array(
                [30, 30, 24, 24, 18, 12, 6, 1]
            )  # Robinson & Stone (2004)
        # Patch option 1, 153 patches, Wallenberg et al. (2022)
        elif patch_option == 2:
            annulino = np.array([0, 12, 24, 36, 48, 60, 72, 84, 90])
            skyvaultaltint = np.array(
                [6, 18, 30, 42, 54, 66, 78, 90]
            )  # Robinson & Stone (2004)
            azistart = np.array([0, 4, 2, 5, 8, 0, 10, 0])  # Fredrik/Nils
            patches_in_band = np.array([31, 30, 28, 24, 19, 13, 7, 1])  # Nils
        # Patch option 2, 306 patches, test
        elif patch_option == 3:
            annulino = np.array([0, 12, 24, 36, 48, 60, 72, 84, 90])
            skyvaultaltint = np.array(
                [6, 18, 30, 42, 54, 66, 78, 90]
            )  # Robinson & Stone (2004)
            azistart = np.array([0, 4, 2, 5, 8, 0, 10, 0])  # Fredrik/Nils
            patches_in_band = np.array(
                [31 * 2, 30 * 2, 28 * 2, 24 * 2, 19 * 2, 13 * 2, 7 * 2, 1]
            )  # Nils
        # Patch option 3, 612 patches, test
        elif patch_option == 4:
            annulino = np.array(
                [0, 4.5, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90]
            )  # Nils
            skyvaultaltint = np.array(
                [3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 90]
            )  # Nils
            patches_in_band = np.array(
                [
                    31 * 2,
                    31 * 2,
                    30 * 2,
                    30 * 2,
                    28 * 2,
                    28 * 2,
                    24 * 2,
                    24 * 2,
                    19 * 2,
                    19 * 2,
                    13 * 2,
                    13 * 2,
                    7 * 2,
                    7 * 2,
                    1,
                ]
            )  # Nils
            azistart = np.array([0, 0, 4, 4, 2, 2, 5, 5, 8, 8, 0, 0, 10, 10, 0])  # Nils

        skyvaultaziint = np.array([360 / patches for patches in patches_in_band])

        for j in range(0, skyvaultaltint.shape[0]):
            for k in range(0, patches_in_band[j]):
                skyvaultalt = np.append(skyvaultalt, skyvaultaltint[j])
                skyvaultazi = np.append(
                    skyvaultazi, k * skyvaultaziint[j] + azistart[j]
                )

        # skyvaultzen = (90 - skyvaultalt) * deg2rad
        # skyvaultalt = skyvaultalt * deg2rad
        # skyvaultazi = skyvaultazi * deg2rad

        return (
            skyvaultalt,
            skyvaultazi,
            annulino,
            skyvaultaltint,
            patches_in_band,
            skyvaultaziint,
        )


class SVFCalculation(GeoCore):
    def __init__(
        self,
        filepath_dsm,
        filepath_veg_cdsm: str = None,
        filepath_veg_tdsm: str = None,
        folderPath: str = './',
        usevegdem: bool = True,
        useWallHeightAspect: bool = False,
    ):
        self.transmitivity_of_light_through_vegetation = 3
        self.filepath_vegCDMS = filepath_veg_cdsm
        self.filepath_vegTDSM = filepath_veg_tdsm

        self.useWallHeightAspect = useWallHeightAspect
        self.filepath_wallAspect = None
        self.filepath_wallHeight = None
        self.filepath_dsm = filepath_dsm
        self.folderPath = folderPath
        self.usevegdem = usevegdem
        self.thread = None
        self.worker = None
        self.vegthread = None
        self.vegworker = None
        self.svftotal = None
        self.vegdsm = None
        self.vegdsm2 = None
        self.svfbu = None
        self.dsm = None
        self.scale = None
        self.steps = 0

    def run(self):
        self.gdal_dsm = gdal.Open(self.filepath_dsm)
        self.dsm = self.gdal_dsm.ReadAsArray().astype(np.float64)
        sizex = self.dsm.shape[0]
        sizey = self.dsm.shape[1]
        geotransform = self.gdal_dsm.GetGeoTransform()
        self.scale = 1 / geotransform[1]

        # response to issue #85
        nd = self.gdal_dsm.GetRasterBand(1).GetNoDataValue()
        self.dsm[self.dsm == nd] = 0.0
        if self.dsm.min() < 0:
            self.dsm = self.dsm + np.abs(self.dsm.min())

        if 250000 < (sizex * sizey) <= 1000000:
            print(
                'Semi lage grid',
                'This process will take a couple of minutes. '
                'Go and make yourself a cup of tea...',
            )

        if 1000000 < (sizex * sizey) <= 4000000:
            print('Large grid', 'This process will take some time. ' 'Go for lunch...')

        if 4000000 < (sizex * sizey) <= 16000000:
            print(
                'Very large grid',
                'This process will take a long time. ' 'Go for lunch and for a walk...',
            )

        if (sizex * sizey) > 16000000:
            print(
                'Huge grid',
                'This process will take a very long time. '
                'Go home for the weekend or consider to tile your grid',
            )

        if self.filepath_vegCDMS:
            self.usevegdem = True
            dataSet = gdal.Open(self.filepath_vegCDMS)
            self.vegdsm = dataSet.ReadAsArray().astype(np.float64)

            vegsizex = self.vegdsm.shape[0]
            vegsizey = self.vegdsm.shape[1]

            if not (vegsizex == sizex) & (vegsizey == sizey):  # &
                print('Error', 'All grids must be of same extent and resolution')
                return

            if self.filepath_vegTDSM:
                dataSet = gdal.Open(self.filepath_vegTDSM)
                self.vegdsm2 = dataSet.ReadAsArray().astype(np.float64)
            else:
                trunkratio = 25.0 / 100.0
                self.vegdsm2 = self.vegdsm * trunkratio

            vegsizex = self.vegdsm2.shape[0]
            vegsizey = self.vegdsm2.shape[1]

            if not (vegsizex == sizex) & (vegsizey == sizey):  # &
                print('Error', 'All grids must be of same extent and resolution')
                return

        else:
            self.vegdsm = self.dsm * 0.0
            self.vegdsm2 = self.dsm * 0.0
            self.usevegdem = False

        if self.useWallHeightAspect:
            self.gdal_wh = gdal.Open(self.filepath_wallHeight)
            self.wheight = self.gdal_wh.ReadAsArray().astype(np.float)
            vhsizex = self.wheight.shape[0]
            vhsizey = self.wheight.shape[1]
            if not (vhsizex == sizex) & (vhsizey == sizey):  # &
                print(None, 'Error', 'All grids must be of same extent and resolution')
                return

            self.gdal_wa = gdal.Open(self.filepath_wallAspect)
            self.waspect = self.gdal_wa.ReadAsArray().astype(np.float)
            vasizex = self.waspect.shape[0]
            vasizey = self.waspect.shape[1]
            if not (vasizex == sizex) & (vasizey == sizey):
                print(None, 'Error', 'All grids must be of same extent and resolution')
                return

        if self.folderPath == 'None':
            print('Error', 'No selected folder')
            return
        else:
            print('startWorker')
            # self.startWorker(self.dsm, self.vegdsm, self.vegdsm2, self.scale, self.usevegdem, self.wheight, self.waspect)
            self.startWorker(
                self.dsm, self.vegdsm, self.vegdsm2, self.scale, self.usevegdem
            )

    def startWorker(self, dsm, vegdem, vegdem2, scale, usevegdem):
        ret = SVFCalculator(dsm, vegdem, vegdem2, scale, usevegdem).run()
        print('ret', ret)
        self.workerFinished(ret)

    def workerFinished(self, ret):
        filename = os.path.join(self.folderPath, 'SkyViewFactor' + '.tif')

        # temporary fix for mac, ISSUE #15
        pf = sys.platform
        if pf == 'darwin' or pf == 'linux2' or pf == 'linux':
            if not os.path.exists(self.folderPath):
                os.makedirs(self.folderPath)

        if ret is not None:
            self.svfbu = ret['svf']
            svfbuE = ret['svfE']
            svfbuS = ret['svfS']
            svfbuW = ret['svfW']
            svfbuN = ret['svfN']

            saveraster(
                self.gdal_dsm, os.path.join(self.folderPath, 'svf.tif'), self.svfbu
            )
            saveraster(self.gdal_dsm, os.path.join(self.folderPath, 'svfE.tif'), svfbuE)
            saveraster(self.gdal_dsm, os.path.join(self.folderPath, 'svfS.tif'), svfbuS)
            saveraster(self.gdal_dsm, os.path.join(self.folderPath, 'svfW.tif'), svfbuW)
            saveraster(self.gdal_dsm, os.path.join(self.folderPath, 'svfN.tif'), svfbuN)

            if os.path.isfile(os.path.join(self.folderPath, 'svfs.zip')):
                os.remove(os.path.join(self.folderPath, 'svfs.zip'))

            zip = zipfile.ZipFile(os.path.join(self.folderPath, 'svfs.zip'), 'a')
            zip.write(os.path.join(self.folderPath, 'svf.tif'), 'svf.tif')
            zip.write(os.path.join(self.folderPath, 'svfE.tif'), 'svfE.tif')
            zip.write(os.path.join(self.folderPath, 'svfS.tif'), 'svfS.tif')
            zip.write(os.path.join(self.folderPath, 'svfW.tif'), 'svfW.tif')
            zip.write(os.path.join(self.folderPath, 'svfN.tif'), 'svfN.tif')
            zip.close()

            os.remove(os.path.join(self.folderPath, 'svf.tif'))
            os.remove(os.path.join(self.folderPath, 'svfE.tif'))
            os.remove(os.path.join(self.folderPath, 'svfS.tif'))
            os.remove(os.path.join(self.folderPath, 'svfW.tif'))
            os.remove(os.path.join(self.folderPath, 'svfN.tif'))

            if not self.usevegdem:
                self.svftotal = self.svfbu
            else:
                # report the result
                svfveg = ret['svfveg']
                svfEveg = ret['svfEveg']
                svfSveg = ret['svfSveg']
                svfWveg = ret['svfWveg']
                svfNveg = ret['svfNveg']
                svfaveg = ret['svfaveg']
                svfEaveg = ret['svfEaveg']
                svfSaveg = ret['svfSaveg']
                svfWaveg = ret['svfWaveg']
                svfNaveg = ret['svfNaveg']

                saveraster(
                    self.gdal_dsm, os.path.join(self.folderPath, 'svfveg.tif'), svfveg
                )
                saveraster(
                    self.gdal_dsm, os.path.join(self.folderPath, 'svfEveg.tif'), svfEveg
                )
                saveraster(
                    self.gdal_dsm, os.path.join(self.folderPath, 'svfSveg.tif'), svfSveg
                )
                saveraster(
                    self.gdal_dsm, os.path.join(self.folderPath, 'svfWveg.tif'), svfWveg
                )
                saveraster(
                    self.gdal_dsm, os.path.join(self.folderPath, 'svfNveg.tif'), svfNveg
                )
                saveraster(
                    self.gdal_dsm, os.path.join(self.folderPath, 'svfaveg.tif'), svfaveg
                )
                saveraster(
                    self.gdal_dsm,
                    os.path.join(self.folderPath, 'svfEaveg.tif'),
                    svfEaveg,
                )
                saveraster(
                    self.gdal_dsm,
                    os.path.join(self.folderPath, 'svfSaveg.tif'),
                    svfSaveg,
                )
                saveraster(
                    self.gdal_dsm,
                    os.path.join(self.folderPath, 'svfWaveg.tif'),
                    svfWaveg,
                )
                saveraster(
                    self.gdal_dsm,
                    os.path.join(self.folderPath, 'svfNaveg.tif'),
                    svfNaveg,
                )

                zip = zipfile.ZipFile(os.path.join(self.folderPath, 'svfs.zip'), 'a')
                zip.write(os.path.join(self.folderPath, 'svfveg.tif'), 'svfveg.tif')
                zip.write(os.path.join(self.folderPath, 'svfEveg.tif'), 'svfEveg.tif')
                zip.write(os.path.join(self.folderPath, 'svfSveg.tif'), 'svfSveg.tif')
                zip.write(os.path.join(self.folderPath, 'svfWveg.tif'), 'svfWveg.tif')
                zip.write(os.path.join(self.folderPath, 'svfNveg.tif'), 'svfNveg.tif')
                zip.write(os.path.join(self.folderPath, 'svfaveg.tif'), 'svfaveg.tif')
                zip.write(os.path.join(self.folderPath, 'svfEaveg.tif'), 'svfEaveg.tif')
                zip.write(os.path.join(self.folderPath, 'svfSaveg.tif'), 'svfSaveg.tif')
                zip.write(os.path.join(self.folderPath, 'svfWaveg.tif'), 'svfWaveg.tif')
                zip.write(os.path.join(self.folderPath, 'svfNaveg.tif'), 'svfNaveg.tif')
                zip.close()

                os.remove(os.path.join(self.folderPath, 'svfveg.tif'))
                os.remove(os.path.join(self.folderPath, 'svfEveg.tif'))
                os.remove(os.path.join(self.folderPath, 'svfSveg.tif'))
                os.remove(os.path.join(self.folderPath, 'svfWveg.tif'))
                os.remove(os.path.join(self.folderPath, 'svfNveg.tif'))
                os.remove(os.path.join(self.folderPath, 'svfaveg.tif'))
                os.remove(os.path.join(self.folderPath, 'svfEaveg.tif'))
                os.remove(os.path.join(self.folderPath, 'svfSaveg.tif'))
                os.remove(os.path.join(self.folderPath, 'svfWaveg.tif'))
                os.remove(os.path.join(self.folderPath, 'svfNaveg.tif'))

                trans = self.transmitivity_of_light_through_vegetation / 100.0

                self.svftotal = self.svfbu - (1 - svfveg) * (1 - trans)

            saveraster(self.gdal_dsm, filename, self.svftotal)

            # Save shadow images for SOLWEIG 2019a
            shmat = ret['shmat']
            vegshmat = ret['vegshmat']
            vbshvegshmat = ret['vbshvegshmat']
            # wallshmat = ret["wallshmat"]
            # wallsunmat = ret["wallsunmat"]
            # wallshvemat = ret["wallshvemat"]
            # facesunmat = ret["facesunmat"]

            np.savez_compressed(
                os.path.join(self.folderPath, 'shadowmats.npz'),
                shadowmat=shmat,
                vegshadowmat=vegshmat,
                vbshmat=vbshvegshmat,
            )  # ,
            # vbshvegshmat=vbshvegshmat, wallshmat=wallshmat, wallsunmat=wallsunmat,
            # facesunmat=facesunmat, wallshvemat=wallshvemat)

        else:
            print(
                'Operations cancelled either by user or error. See the General tab in Log Meassages Panel (speech bubble, lower right) for more information.'
            )


if __name__ == '__main__':
    geocore = GeoCore()
    geocore.bbox = [-1.152704, 46.181627, -1.139893, 46.186990]
    a = SVFCalculation(
        filepath_dsm='/Users/Boris/Documents/TIPEE/pymdu/Tests/umep/DEM.tiff',
        folderPath='.',
    )
    a.run()
