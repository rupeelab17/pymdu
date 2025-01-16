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
import numpy as np
from osgeo import gdal

from pymdu.physics.solar.WallAlgoritms import findwalls
from pymdu.physics.solar.WallAspectWorker import WallAspectWorker


class WallHeightAspect(object):
    def __init__(self, outputFileHeight, outputFileAspect, filepath_dsm, walllimit):
        self.gdal_dsm = None
        self.outputFileHeight = outputFileHeight
        self.outputFileAspect = outputFileAspect
        self.filepath_dsm = filepath_dsm
        self.walllimit = walllimit

    def run(self):
        self.gdal_dsm = gdal.Open(self.filepath_dsm)
        dsm = self.gdal_dsm.ReadAsArray().astype(np.float64)
        geotransform = self.gdal_dsm.GetGeoTransform()
        scale = 1 / geotransform[1]

        walls = findwalls(dsm, self.walllimit)

        # Workaround to avoid NoDataValues
        wallssave = np.copy(walls)

        self.saverasternd(
            gdal_data=self.gdal_dsm, filename=self.outputFileHeight, raster=wallssave
        )

        self.startWorker(wallssave, scale, dsm)

        return {
            'OUTPUT_HEIGHT': self.outputFileHeight,
            'OUTPUT_ASPECT': self.outputFileAspect,
        }

    @staticmethod
    def saverasternd(gdal_data, filename, raster):
        rows = gdal_data.RasterYSize
        cols = gdal_data.RasterXSize

        outDs = gdal.GetDriverByName('GTiff').Create(
            filename, cols, rows, int(1), gdal.GDT_Float32
        )
        outBand = outDs.GetRasterBand(1)

        # write the data
        outBand.WriteArray(raster, 0, 0)
        # flush data to disk, set the NoData value and calculate stats
        outBand.FlushCache()
        # outBand.SetNoDataValue(-9999)

        # georeference the image and set the projection
        outDs.SetGeoTransform(gdal_data.GetGeoTransform())
        outDs.SetProjection(gdal_data.GetProjection())

        outBand = None
        outDs = None

    def startWorker(self, walls, scale, dsm):
        # create a new worker instance
        ret = WallAspectWorker(walls, scale, dsm).run()
        self.workerFinished(ret)

    def workerFinished(self, ret):
        if ret is not None:
            # report the result
            dirwalls = ret['dirwalls']
            # Workaround to avoid NoDataValues
            self.saverasternd(self.gdal_dsm, self.outputFileAspect, dirwalls)
