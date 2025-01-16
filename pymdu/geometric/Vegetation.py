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

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rioxarray

from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.collect.ign.IgnCollect import IgnCollect

try:
    from osgeo import gdal, ogr
except ImportError:
    import gdal
    import ogr


class Vegetation(IgnCollect):
    """
    Class to collect the Vegetation data
    """

    def __init__(
        self,
        filepath_shp: str = None,
        output_path: str = None,
        set_crs: int = None,
        write_file: bool = False,
        min_area: float = 0,
    ):
        """
        Initializes the object with the given parameters.

        Args:
            filepath_shp: (str) The path to the shapefile to be processed. If not provided, the data will be read from the input file.
            output_path (str): The output path for the processed data. If not provided, a default temporary path will be used.
            set_crs: (int) The EPSG code to set the CRS of the output file. If not provided, the CRS of the input file will be used.
            write_file: (bool) If True, the output file will be written to disk. If False, the output file will be returned.
            min_area: (float) The minimum area of the polygons to be considered as vegetation. If not provided, the minimum area will be set to 0.

        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt
            import pymdu.geometric.Vegetation as Vegetation

            plt.clf()  # markdown-exec: hide
            vegetation = Vegetation(output_path='./')
            vegetation.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            vegetation = vegetation.run()
            gdf = vegetation.to_gdf()
            gdf.plot(ax=plt.gca(), edgecolor='black', color='green', alpha=0.5)
            from io import StringIO  # markdown-exec: hide

            buffer = StringIO()  # markdown-exec: hide
            plt.gcf().set_size_inches(10, 5)  # markdown-exec: hide
            plt.savefig(buffer, format='svg', dpi=199)  # markdown-exec: hide
            print(buffer.getvalue())  # markdown-exec: hide
            ```

        Todo:
             * For module TODOs
        """

        super().__init__()
        self.output_path = output_path if output_path else TEMP_PATH
        self.img_tiff_path = os.path.join(TEMP_PATH, 'img.tiff')
        self.ndvi_shp_path = os.path.join(TEMP_PATH, 'ndvi.shp')
        self.ndvi_tif_path = os.path.join(TEMP_PATH, 'ndvi.tif')
        self.output_vegetation_shp_path = os.path.join(TEMP_PATH, 'vegetation.shp')
        self.output_vegetation_geojson_path = os.path.join(
            TEMP_PATH, 'vegetation.geojson'
        )
        self.filepath_shp = filepath_shp
        self.set_crs = set_crs
        self.write_file = write_file
        self.min_area: float = min_area

    def run(self):
        if not self.filepath_shp:
            # if not os.path.exists(self.img_tiff_path):
            self.execute_ign(key='irc')

            dataset = rasterio.open(fp=self.img_tiff_path)
            dataset_rio = rioxarray.open_rasterio(filename=self.img_tiff_path)

            image = dataset.read()
            np.seterr(divide='ignore', invalid='ignore')
            bandNIR = image[0, :, :]
            bandRed = image[1, :, :]
            ndvi = (bandNIR.astype(float) - bandRed.astype(float)) / (
                bandNIR.astype(float) + bandRed.astype(float)
            )
            filter_raster = []
            for x in ndvi:
                filter_raster.append([-999 if y < 0.2 else y for y in x])
            dataset_rio.data[0] = filter_raster
            dataset_rio.rio.to_raster(
                self.ndvi_tif_path,
                tiled=False,
                # GDAL: By default striped TIFF files are created. This option can be used to force creation of tiled TIFF files.
                windowed=True,  # rioxarray: read & write one window at a time
                compress='lzw',
                bigtiff='NO',
                num_threads='all_cpus',
                driver='GTiff',
                predictor=2,
                discard_lsb=2,
            )

            # ================================================
            # open image:
            im = gdal.Open(self.ndvi_tif_path)
            srcband = im.GetRasterBand(1)

            if srcband.GetNoDataValue() is None:
                mask = None
            else:
                mask = srcband

            # create output vector:
            driver = ogr.GetDriverByName('ESRI Shapefile')
            if os.path.exists(self.ndvi_shp_path):
                driver.DeleteDataSource(self.ndvi_shp_path)

            vector = driver.CreateDataSource(self.ndvi_shp_path)
            layer = vector.CreateLayer(
                self.ndvi_shp_path.replace('.shp', ''),
                im.GetSpatialRef(),
                geom_type=ogr.wkbPolygon,
            )

            # create field to write NDVI values:
            field = ogr.FieldDefn('NDVI', ogr.OFTReal)
            layer.CreateField(field)
            del field

            # polygonize:
            gdal.Polygonize(
                srcband, mask, layer, 0, options=[], callback=gdal.TermProgress_nocb
            )

            # close files:
            del im, srcband, vector, layer

            vegetation = gpd.read_file(filename=self.ndvi_shp_path).to_crs(
                epsg=self._epsg
            )
            vegetation = vegetation.loc[(vegetation['NDVI'] == 0)]
            mes_polygons = []
            for x in vegetation['geometry']:
                if x.area > self.min_area:
                    mes_polygons.append(x)
                else:
                    pass
            self.gdf = gpd.GeoDataFrame()
            self.gdf['geometry'] = mes_polygons

        else:
            self.gdf = gpd.read_file(self.filepath_shp, driver='ESRI Shapefile')

        if self.set_crs:
            self.gdf = self.gdf.set_crs(
                crs=self.set_crs, inplace=True, allow_override=True
            )
        else:
            self.gdf.crs = self._epsg

        return self

    def to_gdf(self) -> gpd.GeoDataFrame:
        if self.write_file:
            self.gdf.to_file(self.output_vegetation_shp_path)
            self.gdf.to_file(self.output_vegetation_geojson_path, driver='GeoJSON')
        return self.gdf

    def to_gpkg(self, name: str = 'vegetation'):
        # Write the GeoDataFrame to a GPKG file
        self.gdf.to_file(f'{os.path.join(self.output_path, name)}.gpkg', driver='GPKG')

    def concat_trees_and_vegetation(
        self, name_vegetation: str = 'vegetation.shp', name_trees: str = 'trees.shp'
    ):
        vegetation = gpd.read_file(os.path.join(TEMP_PATH, name_vegetation))
        vegetation['isTree'] = [False for x in vegetation['geometry']]

        trees = gpd.read_file(os.path.join(TEMP_PATH, name_trees))
        trees['isTree'] = [True for x in trees['geometry']]
        trees = trees.to_crs(self._epsg)
        trees['geometry'] = trees.buffer(1, cap_style=3)

        concat = gpd.GeoDataFrame(pd.concat([trees], ignore_index=True))

        # TODO : ajouter la végétation "simplfiée" pour éviter un temps de calcul trop long
        # concat = gpd.GeoDataFrame(pd.concat([trees, vegetation], ignore_index=True))

        concat = concat.to_crs(3857)
        concat['ID_VEG'] = [x for x in concat.index]
        concat['ATTENUATIO'] = [1.9 if x == True else 2.8 for x in concat['isTree']]
        concat['MIN_HEIGHT'] = [2.2 if x == True else 0.4 for x in concat['isTree']]
        concat['MAX_HEIGHT'] = [5.8 if x == True else 1.9 for x in concat['isTree']]

        concat.to_file(os.path.join(TEMP_PATH, 'vegetation.geojson'), driver='GeoJSON')

        return concat


if __name__ == '__main__':
    vegetation = Vegetation(output_path='./')
    vegetation.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    vegetation = vegetation.run()
    vegetation_gdf = vegetation.to_gdf()
    # vegetation.to_shp(name="vegetation")
    import matplotlib.pyplot as plt

    vegetation_gdf.plot(edgecolor='k')
    plt.show()
