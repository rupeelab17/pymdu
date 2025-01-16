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
import os
from datetime import datetime, timedelta

import numpy as np
from osgeo import gdal, osr

from pymdu.GeoCore import GeoCore
from pymdu.physics.solar.DailyShading import dailyshading
from pymdu.physics.solar.UtilitiesSolar import saveraster


class ShadowGenerator(GeoCore):
    def __init__(
        self,
        folderPath,
        filepath_dsm,
        filepath_vegdsm=None,
        filepath_vegdsm_trunk=None,
        filepath_wall_height_layer=None,
        filepath_wall_aspect_layer=None,
        daylight_saving_time: int = 1,
        usevegdem: bool = True,
        usevegdem_trunk: bool = True,
        usewallsh: bool = False,
        trunk_ratio_height: float = 20.0,
        transmissivity=3.0,
        onetime=0,
        date_str: str = '2022-06-21 06:00:00',
        new_lat=None,
        new_lon=None,
    ):
        self.new_lat = new_lat
        self.new_lon = new_lon
        self.folderPath = folderPath
        self.filepath_dsm = filepath_dsm
        self.filepath_wall_height_layer = filepath_wall_height_layer
        self.filepath_vegdsm = filepath_vegdsm
        self.filepath_vegdsm_trunk = filepath_vegdsm_trunk
        self.filepath_wall_aspect_layer = filepath_wall_aspect_layer
        self.usevegdem = usevegdem
        self.usevegdem_trunk = usevegdem_trunk
        self.usewallsh = usewallsh
        self.daylight_saving_time = daylight_saving_time
        self.trunk_ratio_height = trunk_ratio_height
        self.transmissivity = transmissivity / 100.0
        self.year = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').year
        self.month = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').month
        self.day = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S').day
        self.onetime = onetime

    def run(self):
        if self.folderPath == 'None':
            print(None, 'Error', 'Select a valid output folder')
            return

        gdal_dsm = gdal.Open(self.filepath_dsm)
        dsm = gdal_dsm.ReadAsArray().astype(np.float64)

        # response to issue #85
        nd = gdal_dsm.GetRasterBand(1).GetNoDataValue()
        dsm[dsm == nd] = 0.0
        if dsm.min() < 0:
            dsm = dsm + np.abs(dsm.min())

        sizex = dsm.shape[0]
        sizey = dsm.shape[1]

        old_cs = osr.SpatialReference()
        old_cs.ImportFromWkt(gdal_dsm.GetProjectionRef())

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

        new_cs = osr.SpatialReference()
        new_cs.ImportFromWkt(wgs84_wkt)

        transform = osr.CoordinateTransformation(old_cs, new_cs)

        width = gdal_dsm.RasterXSize
        height = gdal_dsm.RasterYSize
        gt = gdal_dsm.GetGeoTransform()
        minx = gt[0]
        miny = gt[3] + width * gt[4] + height * gt[5]
        lonlat = transform.TransformPoint(minx, miny)
        geotransform = gdal_dsm.GetGeoTransform()
        scale = 1 / geotransform[1]

        gdalver = float(gdal.__version__[0])
        if gdalver >= 3.0:
            lon = lonlat[1]  # changed to gdal 3
            lat = lonlat[0]  # changed to gdal 3
        else:
            lon = lonlat[0]  # changed to gdal 2
            lat = lonlat[1]  # changed to gdal 2
        # print('lon:' + str(lon))
        # print('lat:' + str(lat))

        if self.usevegdem:
            dataSet = gdal.Open(self.filepath_vegdsm)
            vegdsm = dataSet.ReadAsArray().astype(np.float64)

            vegsizex = vegdsm.shape[0]
            vegsizey = vegdsm.shape[1]

            if not (vegsizex == sizex) & (vegsizey == sizey):
                print(None, 'Error', 'All grids must be of same extent and resolution')
                return

            if self.usevegdem_trunk:
                # load raster
                gdal.AllRegister()

                dataSet = gdal.Open(self.filepath_vegdsm_trunk)
                vegdsm2 = dataSet.ReadAsArray().astype(np.float64)
            else:
                vegdsm2 = vegdsm * self.trunk_ratio_height

            vegsizex = vegdsm2.shape[0]
            vegsizey = vegdsm2.shape[1]

            if not (vegsizex == sizex) & (vegsizey == sizey):  # &
                print(None, 'Error', 'All grids must be of same extent and resolution')
                return
        else:
            vegdsm = 0
            vegdsm2 = 0

        if self.usewallsh:
            # wall height layer

            self.gdal_wh = gdal.Open(self.filepath_wall_height_layer)
            wheight = self.gdal_wh.ReadAsArray().astype(np.float64)
            vhsizex = wheight.shape[0]
            vhsizey = wheight.shape[1]
            if not (vhsizex == sizex) & (vhsizey == sizey):  # &
                print(None, 'Error', 'All grids must be of same extent and resolution')
                return
            # wall aspectlayer
            if self.filepath_wall_aspect_layer is None:
                print(None, 'Error', 'No valid wall aspect raster layer is selected')
                return

            self.gdal_wa = gdal.Open(self.filepath_wall_aspect_layer)
            waspect = self.gdal_wa.ReadAsArray().astype(np.float64)
            vasizex = waspect.shape[0]
            vasizey = waspect.shape[1]
            if not (vasizex == sizex) & (vasizey == sizey):
                print(None, 'Error', 'All grids must be of same extent and resolution')
                return
        else:
            wheight = 0
            waspect = 0

        if self.folderPath == 'None':
            print(None, 'Error', 'No selected folder')
            return
        else:
            if self.new_lat != None and self.new_lon != None:
                lat = self.new_lat
                lon = self.new_lon
                print('Position has been modified')
            else:
                pass

            UTC = 1
            # if self.dlg.shadowCheckBox.isChecked():
            #     onetime = 1
            #     time = self.dlg.timeEdit.time()
            #     hour = time.hour()
            #     minu = time.minute()
            #     sec = time.second()
            # else:
            #     onetime = 0
            #     hour = 0
            #     minu = 0
            #     sec = 0

            onetime = self.onetime
            hour = 0
            minu = 0
            sec = 0

            tv = [self.year, self.month, self.day, hour, minu, sec]
            timeInterval = timedelta(hours=1).total_seconds() / 60

            # timeInterval = intervalTime.minute() + (intervalTime.hour() * 60) + (intervalTime.second() / 60)
            shadowresult = dailyshading(
                dsm,
                vegdsm,
                vegdsm2,
                scale,
                lon,
                lat,
                sizex,
                sizey,
                tv,
                UTC,
                self.usevegdem,
                timeInterval,
                onetime,
                self.folderPath,
                gdal_dsm,
                self.transmissivity,
                self.daylight_saving_time,
                self.usewallsh,
                wheight,
                waspect,
            )

            shfinal = shadowresult['shfinal']
            time_vector = shadowresult['time_vector']

            if self.onetime == 0:
                timestr = time_vector.strftime('%Y%m%d')
                savestr = 'shadow_fraction_on_'
            else:
                timestr = time_vector.strftime('%Y%m%d_%H%M')
                savestr = 'Shadow_at_'

        filename = os.path.join(self.folderPath, savestr + timestr + '.tif')
        print(filename)

        saveraster(gdal_dsm, filename, shfinal)

        print(None, 'ShadowGenerator', 'Shadow grid(s) successfully generated')


if __name__ == '__main__':
    test = ShadowGenerator(
        folderPath=r'C:\Users\simon\python-scripts\POC/base',
        filepath_dsm=r'C:\Users\simon\python-scripts\POC/DSM.tif',
        usevegdem=False,
        new_lat=None,
        new_lon=None,
    )

    test.run()
