# ******************************************************************************
#  This file is part of pymdu.                                                 *
#                                                                              *
#  Copyright                                                                   *
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
import os
import sys
import zipfile
from datetime import datetime
from shutil import copyfile

import numpy as np
from osgeo import gdal, osr

from pymdu.GeoCore import GeoCore
from pymdu.physics.solar.UtilitiesSolar import saveraster
from pymdu.physics.solar.solweig import WriteMetadataSOLWEIG
from pymdu.physics.solar.solweig.SEBESOLWEIGCommonFiles.Solweig_v2015_metdata_noload import (
    Solweig_2015a_metdata_noload,
)
from pymdu.physics.solar.solweig.SOLWEIGpython.Tgmaps_v1 import Tgmaps_v1
from pymdu.physics.solar.solweig.SolweigWorker import SolweigWorker


class Solweig(GeoCore):
    def __init__(
        self,
        filepath_lancover: str | None = None,
        filepath_veg_cdsm: str | None = None,
        filepath_svf_zip: str | None = None,
        filepath_veg_tdsm: str | None = None,
        filepath_wall_height: str | None = None,
        filepath_wall_aspect: str | None = None,
        filepath_meteo: str | None = None,
        filepath_dsm: str | None = None,
        filepath_dem: str | None = None,
        filepath_shadowmats: str = "shadowmats.npz",
        folderPath: str = "./",
        usevegdem: bool = False,
        useDEM: bool = True,
        usePerez: bool = True,
        useMeteo: bool = False,
        write_other_files: bool = False,
        date_str: str | None = None,
        parallel: bool = True,
    ):
        self.parallel = parallel
        self.meteodata = None
        self.dsm = None
        self.useDEM = useDEM
        self.useMeteo = useMeteo
        self.usePerez = usePerez
        self.date_str = date_str
        self.filePath_shadowmats = filepath_shadowmats
        self.filePath_lancover = filepath_lancover
        self.filePath_cdsm = filepath_veg_cdsm
        self.filePath_tdsm = filepath_veg_tdsm
        self.filePath_dsm = filepath_dsm
        self.filePath_dem = filepath_dem
        self.filePathWallHeight = filepath_wall_height
        self.filePathWallApect = filepath_wall_aspect
        self.write_other_files = write_other_files
        self.folderPath = folderPath
        self.folderPathSVF = filepath_svf_zip
        self.folderPathMeteo = filepath_meteo
        self.usevegdem = usevegdem

        self.trunkHeigh = 25
        self.conifer = False
        self.poi = False
        self.treePlanter = False
        self.onlyGlobal = True
        self.elvis = False
        self.cyl = True
        self.landcover = None
        self.save_trunk = False
        self.gdal_dsm = None
        self.svfbu = None
        self.scale = None
        self.steps = 0
        self.demforbuild = None
        self.lcgrid = None
        self.trans = 3
        self.poisxy = None
        self.poiname = None

    def run(self):
        if self.folderPath is None:
            print("Error", "No selected folder")
            return
        else:
            print("startWorker")
            self.gdal_dsm = gdal.Open(self.filePath_dsm)
            self.dsm = self.gdal_dsm.ReadAsArray().astype(np.float64)
            sizex = self.dsm.shape[0]  # rows
            sizey = self.dsm.shape[1]  # cols
            rows = self.dsm.shape[0]
            cols = self.dsm.shape[1]
            geotransform = self.gdal_dsm.GetGeoTransform()
            self.scale = 1 / geotransform[1]

            alt = np.median(self.dsm)
            if alt < 0:
                alt = 3

            # response to issue #85
            nd = self.gdal_dsm.GetRasterBand(1).GetNoDataValue()
            self.dsm[self.dsm == nd] = 0.0
            # self.dsmcopy = np.copy(self.dsm)
            if self.dsm.min() < 0:
                dsmraise = np.abs(self.dsm.min())
                self.dsm = self.dsm + np.abs(self.dsm.min())
            else:
                dsmraise = 0

            dsm_wkt = self.gdal_dsm.GetProjection()
            dsm_crs = osr.SpatialReference()
            dsm_crs.ImportFromWkt(dsm_wkt)

            wgs84_wkt = """
                        GEOGCS["WGS 84",
                            DATUM["WGS_1984",
                                SPHEROID["WGS 84",6378137,298.257223563,
                                    AUTHORITY["EPSG","7030"]],
                                AUTHORITY["EPSG","6326"]],
                            PRIMEM["Greenwich",0,
                                AUTHORITY["EPSG","8901"]],
                            UNIT["degree",0.01745329251994328,
                                AUTHORITY["EPSG","9122"]],
                            AUTHORITY["EPSG","4326"]]"""

            new_crs = osr.SpatialReference()
            new_crs.ImportFromWkt(wgs84_wkt)

            transform = osr.CoordinateTransformation(dsm_crs, new_crs)

            width = self.gdal_dsm.RasterXSize
            height = self.gdal_dsm.RasterYSize
            minx = geotransform[0]
            miny = geotransform[3] + width * geotransform[4] + height * geotransform[5]
            lonlat = transform.TransformPoint(minx, miny)

            gdalver = float(gdal.__version__[0])
            if gdalver == 3.0:
                lon = lonlat[1]  # changed to gdal 3
                lat = lonlat[0]  # changed to gdal 3
            else:
                lon = lonlat[0]  # changed to gdal 2
                lat = lonlat[1]  # changed to gdal 2

            # UTC = self.dlg.spinBoxUTC.value()
            UTC = 0

            # Vegetation DSMs #
            trunkfile = 0
            trunkratio = 0

            if self.usevegdem:
                self.trans = self.trans / 100.0
                dataSet = gdal.Open(self.filePath_cdsm)
                self.vegdsm = dataSet.ReadAsArray().astype(np.float64)

                vegsizex = self.vegdsm.shape[0]
                vegsizey = self.vegdsm.shape[1]

                if not (vegsizex == sizex) & (vegsizey == sizey):
                    print(
                        "Error in vegetation canopy DSM",
                        "All grids must be of same extent and resolution",
                    )
                    return

                if self.filePath_tdsm:
                    dataSet = gdal.Open(self.filePath_tdsm)
                    self.vegdsm2 = dataSet.ReadAsArray().astype(np.float64)
                    trunkfile = 1
                else:
                    self.filePath_tdsm = None
                    trunkratio = self.trunkHeigh / 100.0
                    self.vegdsm2 = self.vegdsm * trunkratio
                    if self.save_trunk:
                        outDs = gdal.GetDriverByName("GTiff").Create(
                            os.path.join(self.folderPath, "TDSM.tif"),
                            cols,
                            rows,
                            int(1),
                            gdal.GDT_Float32,
                        )
                        outBand = outDs.GetRasterBand(1)
                        outBand.WriteArray(self.vegdsm2, 0, 0)
                        outBand.FlushCache()
                        outDs.SetGeoTransform(self.gdal_dsm.GetGeoTransform())
                        outDs.SetProjection(self.gdal_dsm.GetProjection())
                        # self.saveraster(self.gdal_dsm, self.folderPath[0] + '/TDSM.tif', self.vegdsm2)

                vegsizex = self.vegdsm2.shape[0]
                vegsizey = self.vegdsm2.shape[1]

                if not (vegsizex == sizex) & (vegsizey == sizey):  # &
                    print(
                        "Error in trunk zone DSM",
                        "All grids must be of same extent and resolution",
                    )
                    return

            else:
                self.vegdsm = np.zeros([rows, cols])
                self.vegdsm2 = np.zeros([rows, cols])
                self.usevegdem = False
                self.filePath_cdsm = None
                self.filePath_tdsm = None

            # Land cover #
            if self.filePath_lancover:
                self.landcover = True
                self.demforbuild = 0

                dataSet = gdal.Open(self.filePath_lancover)
                self.lcgrid = dataSet.ReadAsArray().astype(np.float64)

                lcsizex = self.lcgrid.shape[0]
                lcsizey = self.lcgrid.shape[1]

                if not (lcsizex == sizex) & (lcsizey == sizey):
                    print(
                        "Error in land cover grid",
                        "All grids must be of same extent and resolution",
                    )
                    return

                baddataConifer = self.lcgrid == 3
                baddataDecid = self.lcgrid == 4
                if baddataConifer.any():
                    print(
                        "Error in land cover grid",
                        "Land cover grid includes Confier land cover class. "
                        "Ground cover information (underneath canopy) is required.",
                    )
                    return
                if baddataDecid.any():
                    print(
                        "Error in land cover grid",
                        "Land cover grid includes Decidiuous land cover class. "
                        "Ground cover information (underneath canopy) is required.",
                    )
                    return
            else:
                self.filePath_lancover = None

            # DEM #
            if not self.useDEM:
                self.demforbuild = 1
                dataSet = gdal.Open(self.filePath_dem)
                self.dem = dataSet.ReadAsArray().astype(np.float64)

                demsizex = self.dem.shape[0]
                demsizey = self.dem.shape[1]

                if not (demsizex == sizex) & (demsizey == sizey):
                    print(
                        "Error in DEM",
                        "All grids must be of same extent and resolution",
                    )
                    return

                # response to issue #230
                nd = dataSet.GetRasterBand(1).GetNoDataValue()
                self.dem[self.dem == nd] = 0.0
                if self.dem.min() < 0:
                    demraise = np.abs(self.dem.min())
                    self.dem = self.dem + np.abs(self.dem.min())
                else:
                    demraise = 0

                if (dsmraise != demraise) and (dsmraise - demraise > 0.5):
                    print(
                        "WARNiNG! DEM and DSM was raised unequally (difference > 0.5 m). Check your input data!"
                    )

                alt = np.median(self.dem)
                if alt > 0:
                    alt = 3.0

            # SVFs #
            if self.folderPathSVF is None:
                print(
                    "Error",
                    "No SVF zipfile is selected. Use the Sky View Factor"
                    "Calculator to generate svf.zip",
                )
                return
            else:
                zip = zipfile.ZipFile(self.folderPathSVF, "r")
                zip.extractall(self.folderPath)
                zip.close()

                try:
                    dataSet = gdal.Open(os.path.join(self.folderPath, "svf.tif"))
                    svf = dataSet.ReadAsArray().astype(np.float64)
                    dataSet = gdal.Open(os.path.join(self.folderPath, "svfN.tif"))
                    svfN = dataSet.ReadAsArray().astype(np.float64)
                    dataSet = gdal.Open(os.path.join(self.folderPath, "svfS.tif"))
                    svfS = dataSet.ReadAsArray().astype(np.float64)
                    dataSet = gdal.Open(os.path.join(self.folderPath, "svfE.tif"))
                    svfE = dataSet.ReadAsArray().astype(np.float64)
                    dataSet = gdal.Open(os.path.join(self.folderPath, "svfW.tif"))
                    svfW = dataSet.ReadAsArray().astype(np.float64)

                    if self.usevegdem:
                        dataSet = gdal.Open(os.path.join(self.folderPath, "svfveg.tif"))
                        svfveg = dataSet.ReadAsArray().astype(np.float64)
                        dataSet = gdal.Open(
                            os.path.join(self.folderPath, "svfNveg.tif")
                        )
                        svfNveg = dataSet.ReadAsArray().astype(np.float64)
                        dataSet = gdal.Open(
                            os.path.join(self.folderPath, "svfSveg.tif")
                        )
                        svfSveg = dataSet.ReadAsArray().astype(np.float64)
                        dataSet = gdal.Open(
                            os.path.join(self.folderPath, "svfEveg.tif")
                        )
                        svfEveg = dataSet.ReadAsArray().astype(np.float64)
                        dataSet = gdal.Open(
                            os.path.join(self.folderPath, "svfWveg.tif")
                        )
                        svfWveg = dataSet.ReadAsArray().astype(np.float64)

                        dataSet = gdal.Open(
                            os.path.join(self.folderPath, "svfaveg.tif")
                        )
                        svfaveg = dataSet.ReadAsArray().astype(np.float64)
                        dataSet = gdal.Open(
                            os.path.join(self.folderPath, "svfNaveg.tif")
                        )
                        svfNaveg = dataSet.ReadAsArray().astype(np.float64)
                        dataSet = gdal.Open(
                            os.path.join(self.folderPath, "svfSaveg.tif")
                        )
                        svfSaveg = dataSet.ReadAsArray().astype(np.float64)
                        dataSet = gdal.Open(
                            os.path.join(self.folderPath, "svfEaveg.tif")
                        )
                        svfEaveg = dataSet.ReadAsArray().astype(np.float64)
                        dataSet = gdal.Open(
                            os.path.join(self.folderPath, "svfWaveg.tif")
                        )
                        svfWaveg = dataSet.ReadAsArray().astype(np.float64)
                    else:
                        svfveg = np.ones((rows, cols))
                        svfNveg = np.ones((rows, cols))
                        svfSveg = np.ones((rows, cols))
                        svfEveg = np.ones((rows, cols))
                        svfWveg = np.ones((rows, cols))
                        svfaveg = np.ones((rows, cols))
                        svfNaveg = np.ones((rows, cols))
                        svfSaveg = np.ones((rows, cols))
                        svfEaveg = np.ones((rows, cols))
                        svfWaveg = np.ones((rows, cols))
                except:
                    print(
                        "SVF import error",
                        "The zipfile including the SVFs seems corrupt. "
                        "Retry calcualting the SVFs in the Pre-processor or choose "
                        "another file ",
                    )
                    return

                svfsizex = svf.shape[0]
                svfsizey = svf.shape[1]

                if not (svfsizex == sizex) & (svfsizey == sizey):  # &
                    print(
                        "Error in svf rasters",
                        "All grids must be of same extent and resolution",
                    )
                    return

                tmp = svf + svfveg - 1.0
                print("tmp", tmp)
                tmp[tmp < 0.0] = 0.0

                # RUSTINE BORIS
                # tmp[tmp == 1.] = 0.99

                # %matlab crazyness around 0
                svfalfa = np.arcsin(np.exp((np.log((1.0 - tmp)) / 2.0)))

            # Wall height and aspect #
            if self.filePathWallHeight is None:
                print("Error", "No valid wall height grid is selected")
                return

            dataSet = gdal.Open(self.filePathWallHeight)
            wallheight = dataSet.ReadAsArray().astype(np.float64)

            wallheightsizex = wallheight.shape[0]
            wallheightsizey = wallheight.shape[1]

            if not (wallheightsizex == sizex) & (wallheightsizey == sizey):
                print(
                    "Error in wall height grid",
                    "All grids must be of same extent and resolution",
                )
                return

            if self.filePathWallApect is None:
                print("Error", "No valid wall aspect grid is selected")
                return

            dataSet = gdal.Open(self.filePathWallApect)
            wallaspect = dataSet.ReadAsArray().astype(np.float64)

            wallaspectsizex = wallaspect.shape[0]
            wallaspectsizey = wallaspect.shape[1]

            if not (wallaspectsizex == sizex) & (wallaspectsizey == sizey):
                print(
                    "Error in wall aspect grid",
                    "All grids must be of same extent and resolution",
                )
                return

            if (sizex * sizey) > 250000 and (sizex * sizey) <= 1000000:
                print("Semi lage grid", "This process will take a couple of minutes.")

            if (sizex * sizey) > 1000000 and (sizex * sizey) <= 4000000:
                print("Large grid", "This process will take some time.")

            if (sizex * sizey) > 4000000 and (sizex * sizey) <= 16000000:
                print("Very large grid", "This process will take a long time.")

            if (sizex * sizey) > 16000000:
                print("Huge grid", "This process will take a very long time.")

            # Meteorological data #
            Twater = []
            if self.useMeteo:
                self.meteodata = self.read_meteodata()
                metfileexist = 1
                self.PathMet = self.folderPathMeteo
            else:
                metfileexist = 0
                self.PathMet = None
                self.meteodata = np.zeros((1, 24)) - 999.0

                # date = self.dlg.calendarWidget.selectedDate()
                date = datetime.strptime(self.date_str, "%Y-%m-%d %H:%M:%S")
                year = date.year
                month = date.month
                day = date.day

                # time = self.dlg.spinBoxTimeEdit.time()
                hour = date.hour
                minu = date.minute
                doy = self.day_of_year(year, month, day)

                Ta = 30
                RH = 10
                radG = 10
                radD = 10
                radI = 10
                Twater = 10
                Ws = 10

                self.meteodata[0, 0] = year
                self.meteodata[0, 1] = doy
                self.meteodata[0, 2] = hour
                self.meteodata[0, 3] = minu
                self.meteodata[0, 11] = Ta
                self.meteodata[0, 10] = RH
                self.meteodata[0, 14] = radG
                self.meteodata[0, 21] = radD
                self.meteodata[0, 22] = radI
                self.meteodata[0, 9] = Ws

            # Other parameters #
            # ShortwaveHuman
            absK = 0.7
            # LongwaveHuman
            absL = 0.95
            # posture
            pos = 1

            if self.cyl:
                cyl = 1
            else:
                cyl = 0

            if pos == 0:
                Fside = 0.22
                Fup = 0.06
                height = 1.1
                Fcyl = 0.28
            else:
                Fside = 0.166666
                Fup = 0.166666
                height = 0.75
                Fcyl = 0.2

            albedo_b = 0.2
            albedo_g = 0.15
            ewall = 0.9
            eground = 0.95

            if self.elvis:
                elvis = 1
            else:
                elvis = 0

            # %Initialization of maps
            Knight = np.zeros((rows, cols))
            Tgmap1 = np.zeros((rows, cols))
            Tgmap1E = np.zeros((rows, cols))
            Tgmap1S = np.zeros((rows, cols))
            Tgmap1W = np.zeros((rows, cols))
            Tgmap1N = np.zeros((rows, cols))

            # building grid and land cover preparation
            # sitein = os.path.join(".", "solveig/landcoverclasses_2016a.txt")
            sitein = str(
                self.physics_path.joinpath("solar/solweig/landcoverclasses_2016a.txt")
            )

            f = open(sitein, "r")
            lin = f.readlines()
            lc_class = np.zeros((lin.__len__() - 1, 6))
            for i in range(1, lin.__len__()):
                lines = lin[i].split()
                for j in np.arange(1, 7):
                    lc_class[i - 1, j - 1] = float(lines[j])
            f.close()

            if self.useDEM:
                buildings = np.copy(self.lcgrid)
                buildings[buildings == 7] = 1
                buildings[buildings == 6] = 1
                buildings[buildings == 5] = 1
                buildings[buildings == 4] = 1
                buildings[buildings == 3] = 1
                buildings[buildings == 2] = 0
            else:
                buildings = self.dsm - self.dem
                buildings[buildings < 2.0] = 1.0
                buildings[buildings >= 2.0] = 0.0

            saveraster(
                self.gdal_dsm, os.path.join(self.folderPath, "buildings.tif"), buildings
            )

            if self.onlyGlobal:
                onlyglobal = 1
            else:
                onlyglobal = 0

            location = {"longitude": lon, "latitude": lat, "altitude": alt}
            YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax = (
                Solweig_2015a_metdata_noload(
                    inputdata=self.meteodata, location=location, UTC=UTC
                )
            )

            # %Creating vectors from meteorological input
            DOY = self.meteodata[:, 1]
            hours = self.meteodata[:, 2]
            minu = self.meteodata[:, 3]
            Ta = self.meteodata[:, 11]
            RH = self.meteodata[:, 10]
            radG = self.meteodata[:, 14]
            radD = self.meteodata[:, 21]
            radI = self.meteodata[:, 22]
            P = self.meteodata[:, 12]
            Ws = self.meteodata[:, 9]
            # %Wd=met(:,13);

            if self.treePlanter:
                treeplanter = 1
                if metfileexist == 0:
                    print(
                        "Meteorological file missing",
                        "To generate data for the TreePlanter, a meteorological "
                        "input file must be used.",
                    )
                    return
            else:
                treeplanter = 0

            # Check if diffuse and direct radiation exist
            if metfileexist == 1:
                if onlyglobal == 0:
                    if np.min(radD) == -999:
                        print(
                            "Diffuse radiation include NoData values (-999)",
                            'Tick in the box "Estimate diffuse and direct shortwave..." or aqcuire '
                            "observed values from external data sources.",
                        )
                        return
                    if np.min(radI) == -999:
                        print(
                            "Direct radiation include NoData values (-999)",
                            'Tick in the box "Estimate diffuse and direct shortwave..." or aqcuire '
                            "observed values from external data sources.",
                        )
                        return

            # POIs check
            # if self.poi:
            #     header = 'yyyy id   it imin dectime altitude azimuth kdir kdiff kglobal kdown   kup    keast ksouth ' \
            #              'kwest knorth ldown   lup    least lsouth lwest  lnorth   Ta      Tg     RH    Esky   Tmrt    ' \
            #              'I0     CI   Shadow  SVF_b  SVF_bv KsideI PET UTCI'
            #
            #     poilyr = self.layerComboManagerPOI.currentLayer()
            #     if poilyr is None:
            #         QMessageBox.critical(self.dlg, "Error", "No valid point layer is selected")
            #         return
            #
            #     poi_field = self.layerComboManagerPOIfield.currentField()
            #     if poi_field is None:
            #         QMessageBox.critical(self.dlg, "Error", "An attribute with unique values must be selected")
            #         return
            #
            #     if metfileexist == 1:
            #         if np.min(Ws) == -999:
            #             QMessageBox.critical(self.dlg, "Wind speed include NoData values (-999)",
            #                                  'Wind speed is required to calculate PET and UTCI at the POIs')
            #             return
            #
            #     vlayer = QgsVectorLayer(poilyr.source(), "point", "ogr")
            #     idx = vlayer.fields().indexFromName(poi_field)
            #     numfeat = vlayer.featureCount()
            #     self.poiname = []
            #     self.poisxy = np.zeros((numfeat, 3)) - 999
            #     ind = 0
            #     for f in vlayer.getFeatures():  # looping through each POI
            #         y = f.geometry().centroid().asPoint().y()
            #         x = f.geometry().centroid().asPoint().x()
            #
            #         self.poiname.append(f.attributes()[idx])
            #         self.poisxy[ind, 0] = ind
            #         self.poisxy[ind, 1] = np.round((x - minx) * self.scale)
            #         if miny >= 0:
            #             self.poisxy[ind, 2] = np.round((miny + rows * (1. / self.scale) - y) * self.scale)
            #         else:
            #             self.poisxy[ind, 2] = np.round((miny + rows * (1. / self.scale) - y) * self.scale)
            #
            #         ind += 1
            #
            #     uni = set(self.poiname)
            #     if not uni.__len__() == self.poisxy.shape[0]:
            #         QMessageBox.critical(self.dlg, "Error", "A POI attribute with unique values must be selected")
            #         return
            #
            #     for k in range(0, self.poisxy.shape[0]):
            #         poi_save = []  # np.zeros((1, 33))
            #         data_out = self.folderPath[0] + '/POI_' + str(self.poiname[k]) + '.txt'
            #         np.savetxt(data_out, poi_save, delimiter=' ', header=header, comments='')  # fmt=numformat,

            # %Parameterisarion for Lup
            if not height:
                height = 1.1

            # %Radiative surface influence, Rule of thumb by Schmid et al. (1990).
            first = np.round(height)
            if first == 0.0:
                first = 1.0
            second = np.round((height * 20.0))

            if self.usevegdem:
                # Conifer or deciduous
                if self.conifer:
                    # If conifer, "leaves" all year
                    leafon = np.ones((1, DOY.shape[0]))
                else:
                    # If deciduous, leaves part of year
                    # firstdayleaf = self.dlg.spinBoxFirstDay.value()
                    # lastdayleaf = self.dlg.spinBoxLastDay.value()
                    firstdayleaf = 97
                    lastdayleaf = 300

                    leafon = np.zeros((1, DOY.shape[0]))
                    if firstdayleaf > lastdayleaf:
                        # Southern hemisphere?
                        leaf_bool = (DOY > firstdayleaf) | (DOY < lastdayleaf)
                    else:
                        # Northern hemisphere?
                        leaf_bool = (DOY > firstdayleaf) & (DOY < lastdayleaf)
                    leafon[0, leaf_bool] = 1

                    # % Vegetation transmittivity of shortwave radiation
                psi = leafon * self.trans
                psi[leafon == 0] = 0.5
                # amaxvalue
                vegmax = self.vegdsm.max()
                amaxvalue = self.dsm.max() - self.dsm.min()
                amaxvalue = np.maximum(amaxvalue, vegmax)

                # Elevation vegdsms if buildingDEM includes ground heights
                self.vegdsm = self.vegdsm + self.dsm
                self.vegdsm[self.vegdsm == self.dsm] = 0
                self.vegdsm2 = self.vegdsm2 + self.dsm
                self.vegdsm2[self.vegdsm2 == self.dsm] = 0

                # % Bush separation
                bush = np.logical_not((self.vegdsm2 * self.vegdsm)) * self.vegdsm

                svfbuveg = svf - (1.0 - svfveg) * (
                    1.0 - self.trans
                )  # % major bug fixed 20141203
            else:
                psi = leafon * 0.0 + 1.0
                svfbuveg = svf
                bush = np.zeros([rows, cols])
                amaxvalue = 0

            # Import shadow matrices (Anisotropic sky)
            if self.usePerez:
                if self.filePath_shadowmats is None:
                    print(
                        "Error",
                        "No Shadow file is selected. Use the Sky View Factor"
                        "Calculator to generate shadowmats.npz",
                    )
                    return
                else:
                    anisotropic_sky = 1
                    data = np.load(self.filePath_shadowmats)
                    shmat = data["shadowmat"]
                    vegshmat = data["vegshadowmat"]
                    vbshvegshmat = data["vbshmat"]
                    if self.usevegdem:
                        diffsh = np.zeros((rows, cols, shmat.shape[2]))
                        for i in range(0, shmat.shape[2]):
                            diffsh[:, :, i] = shmat[:, :, i] - (
                                1 - vegshmat[:, :, i]
                            ) * (
                                1 - self.trans
                            )  # changes in psi not implemented yet
                    else:
                        diffsh = shmat
                        vegshmat += 1
                        vbshvegshmat += 1

                    # Estimate number of patches based on shadow matrices
                    if shmat.shape[2] == 145:
                        patch_option = 1  # patch_option = 1 # 145 patches
                    elif shmat.shape[2] == 153:
                        patch_option = 2  # patch_option = 2 # 153 patches
                    elif shmat.shape[2] == 306:
                        patch_option = 3  # patch_option = 3 # 306 patches
                    elif shmat.shape[2] == 612:
                        patch_option = 4  # patch_option = 4 # 612 patches

                    # asvf to calculate sunlit and shaded patches
                    asvf = np.arccos(np.sqrt(svf))

            else:
                anisotropic_sky = 0
                diffsh = None
                shmat = None
                vegshmat = None
                vbshvegshmat = None
                asvf = None
                patch_option = 0

            # % Ts parameterisation maps
            if self.landcover:
                if np.max(self.lcgrid) > 7 or np.min(self.lcgrid) < 1:
                    print(
                        "Attention! The land cover grid includes integer values higher (or lower) than UMEP-formatted land cover grid (should be integer between 1 and 7). If other LC-classes should be included they also need to be included in landcoverclasses_2016a.txt"
                    )
                    return
                    # QMessageBox.critical(self.dlg, "Error", "The land cover grid includes values not appropriate for UMEP-formatted land cover grid (should be integer between 1 and 7).")
                    # return
                if np.where(self.lcgrid) == 3 or np.where(self.lcgrid) == 4:
                    print(
                        "Error The land cover grid includes values (decidouos and/or conifer) not appropriate for SOLWEIG-formatted land cover grid (should not include 3 or 4)."
                    )
                    return
                [
                    TgK,
                    Tstart,
                    alb_grid,
                    emis_grid,
                    TgK_wall,
                    Tstart_wall,
                    TmaxLST,
                    TmaxLST_wall,
                ] = Tgmaps_v1(self.lcgrid, lc_class)
            else:
                TgK = Knight + 0.37
                Tstart = Knight - 3.41
                alb_grid = Knight + albedo_g
                emis_grid = Knight + eground
                TgK_wall = 0.37
                Tstart_wall = -3.41
                TmaxLST = 15.0
                TmaxLST_wall = 15.0

            # Initialisation of time related variables
            if Ta.__len__() == 1:
                timestepdec = 0
            else:
                timestepdec = dectime[1] - dectime[0]
            timeadd = 0.0
            timeaddE = 0.0
            timeaddS = 0.0
            timeaddW = 0.0
            timeaddN = 0.0
            firstdaytime = 1.0

            WriteMetadataSOLWEIG.writeRunInfo(
                self.folderPath,
                self.filePath_dsm,
                self.gdal_dsm,
                self.usevegdem,
                self.filePath_cdsm,
                trunkfile,
                self.filePath_tdsm,
                lat,
                lon,
                UTC,
                self.landcover,
                self.filePath_lancover,
                metfileexist,
                self.PathMet,
                self.meteodata,
                ".",
                absK,
                absL,
                albedo_b,
                albedo_g,
                ewall,
                eground,
                onlyglobal,
                trunkratio,
                self.trans,
                rows,
                cols,
                pos,
                elvis,
                cyl,
                self.demforbuild,
                anisotropic_sky,
                treeplanter,
            )

            # Save files for Tree Planter
            if self.treePlanter:
                # Save DSM
                copyfile(self.filePath_dsm, os.path.join(self.folderPath, "DSM.tif"))

                # Save met file
                copyfile(self.PathMet, os.path.join(self.folderPath, "metfile.txt"))

                # Save CDSM
                if self.usevegdem:
                    copyfile(
                        self.filePath_cdsm, os.path.join(self.folderPath, "CDSM.tif")
                    )

                # Saving settings from SOLWEIG for SOLWEIG1D in TreePlanter
                settingsHeader = (
                    "UTC, posture, onlyglobal, landcover, anisotropic, cylinder, albedo_walls, "
                    "albedo_ground, emissivity_walls, emissivity_ground, absK, absL, elevation, "
                    "patch_option"
                )
                settingsFmt = (
                    "%i",
                    "%i",
                    "%i",
                    "%i",
                    "%i",
                    "%i",
                    "%1.2f",
                    "%1.2f",
                    "%1.2f",
                    "%1.2f",
                    "%1.2f",
                    "%1.2f",
                    "%1.2f",
                    "%i",
                )
                settingsData = np.array(
                    [
                        [
                            UTC,
                            pos,
                            onlyglobal,
                            self.landcover,
                            anisotropic_sky,
                            cyl,
                            albedo_b,
                            albedo_g,
                            ewall,
                            eground,
                            absK,
                            absL,
                            alt,
                            patch_option,
                        ]
                    ]
                )
                np.savetxt(
                    os.path.join(self.folderPath, "treeplantersettings.txt"),
                    settingsData,
                    fmt=settingsFmt,
                    header=settingsHeader,
                    delimiter=" ",
                )

            #  If metfile starts at night
            CI = 1.0

            # PET variables
            # mbody = self.dlg.doubleSpinBoxWeight.value()
            # ht = self.dlg.doubleSpinBoxHeight.value() / 100.
            # clo = self.dlg.doubleSpinBoxClo.value()
            # age = self.dlg.doubleSpinBoxAge.value()
            # activity = self.dlg.doubleSpinBoxActivity.value()
            # sex = self.dlg.comboBoxGender.currentIndex() + 1
            # sensorheight = self.dlg.doubleSpinBoxWsHt.value()

            # PET variables
            # Weight kg
            mbody = 75
            # Height cm
            ht = 180 / 100.0
            # Clothing
            clo = 0.9
            # Age
            age = 35
            # Activity
            activity = 80
            # Gender
            sex = 0 + 1  # sex: 1=male 2=female
            # WsHt height of wind sensor
            sensorheight = 10

            self.startWorker(
                self.dsm,
                self.scale,
                rows,
                cols,
                svf,
                svfN,
                svfW,
                svfE,
                svfS,
                svfveg,
                svfNveg,
                svfEveg,
                svfSveg,
                svfWveg,
                svfaveg,
                svfEaveg,
                svfSaveg,
                svfWaveg,
                svfNaveg,
                self.vegdsm,
                self.vegdsm2,
                albedo_b,
                absK,
                absL,
                ewall,
                Fside,
                Fup,
                Fcyl,
                altitude,
                azimuth,
                zen,
                jday,
                self.usevegdem,
                onlyglobal,
                buildings,
                location,
                psi,
                self.landcover,
                self.lcgrid,
                dectime,
                altmax,
                wallaspect,
                wallheight,
                cyl,
                elvis,
                Ta,
                RH,
                radG,
                radD,
                radI,
                P,
                amaxvalue,
                bush,
                Twater,
                TgK,
                Tstart,
                alb_grid,
                emis_grid,
                TgK_wall,
                Tstart_wall,
                TmaxLST,
                TmaxLST_wall,
                first,
                second,
                svfalfa,
                svfbuveg,
                firstdaytime,
                timeadd,
                timeaddE,
                timeaddS,
                timeaddW,
                timeaddN,
                timestepdec,
                Tgmap1,
                Tgmap1E,
                Tgmap1S,
                Tgmap1W,
                Tgmap1N,
                CI,
                YYYY,
                DOY,
                hours,
                minu,
                self.gdal_dsm,
                self.folderPath,
                self.poisxy,
                self.poiname,
                Ws,
                mbody,
                age,
                ht,
                activity,
                clo,
                sex,
                sensorheight,
                diffsh,
                shmat,
                vegshmat,
                vbshvegshmat,
                anisotropic_sky,
                asvf,
                patch_option,
                self.write_other_files,
            )

    def day_of_year(self, yyyy, month, day):
        if (yyyy % 4) == 0:
            if (yyyy % 100) == 0:
                if (yyyy % 400) == 0:
                    leapyear = 1
                else:
                    leapyear = 0
            else:
                leapyear = 1
        else:
            leapyear = 0

        if leapyear == 1:
            dayspermonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        doy = np.sum(dayspermonth[0 : month - 1]) + day

        return doy

    def read_meteodata(self):
        headernum = 1
        delim = " "
        try:
            metdata = np.loadtxt(
                self.folderPathMeteo, skiprows=headernum, delimiter=delim
            )
        except:
            print(
                "Import Error",
                "Make sure format of meteorological file is correct. You can "
                "prepare your data by using 'Prepare Existing Data' in "
                "the Pre-processor",
            )
            return

        if metdata.shape[1] == 24:
            print("SOLWEIG", "Meteorological data succesfully loaded")
        else:
            print(
                "Import Error",
                "Wrong number of columns in meteorological data. You can "
                "prepare your data by using 'Prepare Existing Data' in "
                "the Pre-processor",
            )
            return

        return metdata

    def startWorker(
        self,
        dsm,
        scale,
        rows,
        cols,
        svf,
        svfN,
        svfW,
        svfE,
        svfS,
        svfveg,
        svfNveg,
        svfEveg,
        svfSveg,
        svfWveg,
        svfaveg,
        svfEaveg,
        svfSaveg,
        svfWaveg,
        svfNaveg,
        vegdsm,
        vegdsm2,
        albedo_b,
        absK,
        absL,
        ewall,
        Fside,
        Fup,
        Fcyl,
        altitude,
        azimuth,
        zen,
        jday,
        usevegdem,
        onlyglobal,
        buildings,
        location,
        psi,
        landcover,
        lcgrid,
        dectime,
        altmax,
        wallaspect,
        wallheight,
        cyl,
        elvis,
        Ta,
        RH,
        radG,
        radD,
        radI,
        P,
        amaxvalue,
        bush,
        Twater,
        TgK,
        Tstart,
        alb_grid,
        emis_grid,
        TgK_wall,
        Tstart_wall,
        TmaxLST,
        TmaxLST_wall,
        first,
        second,
        svfalfa,
        svfbuveg,
        firstdaytime,
        timeadd,
        timeaddE,
        timeaddS,
        timeaddW,
        timeaddN,
        timestepdec,
        Tgmap1,
        Tgmap1E,
        Tgmap1S,
        Tgmap1W,
        Tgmap1N,
        CI,
        YYYY,
        DOY,
        hours,
        minu,
        gdal_dsm,
        folderPath,
        poisxy,
        poiname,
        Ws,
        mbody,
        age,
        ht,
        activity,
        clo,
        sex,
        sensorheight,
        diffsh,
        shmat,
        vegshmat,
        vbshvegshmat,
        anisotropic_sky,
        asvf,
        patch_option,
        write_other_files,
    ):
        # create a new worker instance
        ret = SolweigWorker(
            dsm,
            scale,
            rows,
            cols,
            svf,
            svfN,
            svfW,
            svfE,
            svfS,
            svfveg,
            svfNveg,
            svfEveg,
            svfSveg,
            svfWveg,
            svfaveg,
            svfEaveg,
            svfSaveg,
            svfWaveg,
            svfNaveg,
            vegdsm,
            vegdsm2,
            albedo_b,
            absK,
            absL,
            ewall,
            Fside,
            Fup,
            Fcyl,
            altitude,
            azimuth,
            zen,
            jday,
            usevegdem,
            onlyglobal,
            buildings,
            location,
            psi,
            landcover,
            lcgrid,
            dectime,
            altmax,
            wallaspect,
            wallheight,
            cyl,
            elvis,
            Ta,
            RH,
            radG,
            radD,
            radI,
            P,
            amaxvalue,
            bush,
            Twater,
            TgK,
            Tstart,
            alb_grid,
            emis_grid,
            TgK_wall,
            Tstart_wall,
            TmaxLST,
            TmaxLST_wall,
            first,
            second,
            svfalfa,
            svfbuveg,
            firstdaytime,
            timeadd,
            timeaddE,
            timeaddS,
            timeaddW,
            timeaddN,
            timestepdec,
            Tgmap1,
            Tgmap1E,
            Tgmap1S,
            Tgmap1W,
            Tgmap1N,
            CI,
            YYYY,
            DOY,
            hours,
            minu,
            gdal_dsm,
            folderPath,
            poisxy,
            poiname,
            Ws,
            mbody,
            age,
            ht,
            activity,
            clo,
            sex,
            sensorheight,
            diffsh,
            shmat,
            vegshmat,
            vbshvegshmat,
            anisotropic_sky,
            asvf,
            patch_option,
            write_other_files,
        ).run(parallel=self.parallel)
        print("ret", ret)
        self.workerFinished(ret)

    def workerFinished(self, ret):
        filename = os.path.join(self.folderPath, "Tmrt_average" + ".tif")

        # temporary fix for mac, ISSUE #15
        pf = sys.platform
        if pf == "darwin" or pf == "linux2" or pf == "linux":
            if not os.path.exists(self.folderPath):
                os.makedirs(self.folderPath)

        if ret is not None:
            tmrtplot = ret["tmrtplot"]
            saveraster(self.gdal_dsm, filename, tmrtplot)


if __name__ == "__main__":
    geocore = GeoCore()
    geocore.bbox = [-1.152704, 46.181627, -1.139893, 46.186990]
    a = Solweig(
        filepath_dsm="/Users/Boris/Documents/TIPEE/pymdu/Tests/umep/DEM.tiff",
        folderPath=".",
    )
    a.run()
