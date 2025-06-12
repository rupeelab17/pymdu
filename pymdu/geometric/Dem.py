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
from osgeo import gdal
from rasterio.enums import Resampling
from shapely.geometry import box

from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.collect.ign.IgnCollect import IgnCollect


class Dem(IgnCollect):
    """
    Class to collect the Dem data

    """

    def __init__(self, output_path: str | None = None):
        """
        Initializes the object with the given parameters.

        Args:
            output_path (str): The output path for the processed data. If not provided, a default temporary path will be used.
        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt

            plt.clf()  # markdown-exec: hide
            import pymdu.geometric.Dem as Dem
            import rasterio
            import rasterio.plot

            dem = Dem(output_path='./')
            dem.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            ign_dem = dem.run()
            fig, ax = plt.subplots(figsize=(15, 15))
            raster = rasterio.open('DEM.tif')
            rasterio.plot.show(raster, ax=ax, cmap='viridis')
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
        self.dataarray = None
        self.output_path = output_path if output_path else TEMP_PATH
        # self.path_save_tiff_before_manip = os.path.join(self.output_path, 'DEM_before.tif')
        self.path_save_tiff = os.path.join(self.output_path, "DEM.tif")
        self.path_save_mask = os.path.join(self.output_path, "mask.shp")
        self.path_temp_tiff = os.path.join(TEMP_PATH, "dem.tiff")

        if os.path.exists(self.path_temp_tiff):
            os.remove(self.path_temp_tiff)
        if os.path.exists(self.path_save_tiff):
            os.remove(self.path_save_tiff)

    def run(self, shape: tuple = None):
        self.content = self.execute_ign(key="dem").content

        import rioxarray as rxr

        dataarray = rxr.open_rasterio(self.path_temp_tiff)
        # scaled_data = ((dataarray / dataarray.max()) * 255).astype('uint8')
        if shape:
            self.dataarray = dataarray.rio.reproject(
                dst_crs=self._epsg,
                shape=shape,
                resolution=None,
                resampling=Resampling.nearest,  # nodata=-9999, fill_value=-9999
            )
        else:
            self.dataarray = dataarray.rio.reproject(
                dst_crs=self._epsg,
                resolution=1,
                resampling=Resampling.nearest,
                # nodata=-9999, fill_value=-9999
            )
        # Scale the data to 8-bit range (0-255)
        # Update metadata for the output file
        # profile = dataarray.profile
        # profile.update(dtype='uint8', compress='lzw')

        # TODO : j'observe un bug ici
        # CPLE_AppDefinedError: Deleting C:/Users/simon/AppData/Local/Temp/DEM.tiff failed: Permission denied
        try:
            self.dataarray.rio.to_raster(
                self.path_save_tiff,
                compress="lzw",
                bigtiff="YES",
                num_threads="all_cpus",
                tiled=True,
                driver="GTiff",
                predictor=2,
                discard_lsb=2,
            )
        except Exception as e:
            print("Oups Exception ", e)
            print("Oups Exception ", self.path_save_tiff)
            self.dataarray.rio.to_raster(
                "Dem.tif",
                compress="lzw",
                bigtiff="NO",
                num_threads="all_cpus",
                tiled=True,
                driver="GTiff",
                predictor=2,
                discard_lsb=2,
            )

        self.__generate_mask_and_adapt_dem()

        return self

    def content(self):
        return self.content

    # def generate_mask_obsolete(self):
    #     dataarray = rxr.open_rasterio(self.path_save_tiff)
    #     # dataarray = xr.open_rasterio(self.save_path)
    #     df = dataarray[0].to_pandas()
    #
    #     miny = df[df.columns[0]][df[df.columns[0]] > 0].index[-1]
    #     maxy = df[df.columns[-1]][df[df.columns[-1]] > 0].index[0]
    #     minx = df[df.index == df.index[0]].T[df[df.index == df.index[0]].T > 0].dropna().index[1]
    #     maxx = df[df.index == df.index[-1]].T[df[df.index == df.index[-1]].T > 0].dropna().index[0]
    #     self.new_bbox = box(minx, miny, maxx, maxy)
    #
    #     self.gdf = gpd.GeoDataFrame(index=[0], crs='epsg:2154', geometry=[self.new_bbox])
    #     self.gdf.to_file(filename=self.path_save_mask, driver='ESRI Shapefile')
    #
    #     return self

    def __generate_mask_and_adapt_dem(self):
        gdf_project = gpd.GeoDataFrame(
            gpd.GeoSeries(box(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3])),
            columns=["geometry"],
            crs="epsg:4326",
        )
        gdf_project = gdf_project.to_crs(epsg=2154)
        envelope_polygon = gdf_project.envelope.bounds
        bbox = envelope_polygon.values[0]
        bbox_final = box(bbox[0], bbox[1], bbox[2], bbox[3])
        gdf_bbox_mask_2154 = gpd.GeoDataFrame(
            gpd.GeoSeries(bbox_final), columns=["geometry"], crs="epsg:2154"
        )
        gdf_bbox_mask_2154.scale(xfact=0.85, yfact=0.85).to_file(
            self.path_save_mask, driver="ESRI Shapefile"
        )

    def convert_16bit_to_8bit(self, input_path, output_path):
        # Open the input GeoTIFF file
        ds = gdal.Open(input_path)

        # Read the data
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()

        # Scale the data to 8-bit range (0-255) for display
        scaled_data = ((data / data.max()) * 255).astype("uint8")

        # Create a new 8-bit GeoTIFF file
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            output_path, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte
        )
        out_ds.SetProjection(ds.GetProjection())
        out_ds.SetGeoTransform(ds.GetGeoTransform())

        # Write the scaled data to the output GeoTIFF file
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(scaled_data)

        # Close the datasets
        ds = None
        out_ds = None

    def to_shp(self):
        pass


if __name__ == "__main__":
    dem = Dem(output_path="./")
    dem.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    ign_dem = dem.run()

    # dem.convert_16bit_to_8bit("./DEM.tif", "./DEM_8bit.tif")
    # from pymdu.image.geotiff import tif_to_geojson
    import matplotlib.pyplot as plt
    import rasterio
    import rasterio.plot

    # tif_to_geojson("./DEM.tif", "./DEM.geojson")

    fig, ax = plt.subplots(figsize=(15, 15))
    raster = rasterio.open("DEM.tif")
    rasterio.plot.show(raster, ax=ax)
    plt.show()
