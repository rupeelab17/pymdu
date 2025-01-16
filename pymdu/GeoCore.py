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
import glob
import json
import os
from pathlib import Path

import geopandas as gpd
import gitlab
import h3
import matplotlib.pyplot as plt
import orjson as json
import rasterio
from jsonmerge import Merger
from shapely.geometry import box

from pymdu.collect.GlobalVariables import BBOX_LR
from pymdu.commons.BasicFunctions import (
    geo_boundary_to_polygon,
    BasicFunctions,
    geo_lat_lon_from_h3,
)


class classproperty(property):
    def __get__(self, obj, objtype=None):
        return super(classproperty, self).__get__(objtype)

    def __set__(self, obj, value):
        super(classproperty, self).__set__(type(obj), value)

    def __delete__(self, obj):
        super(classproperty, self).__delete__(type(obj))


class GeoCore:
    """
    classdocs
    """

    _bbox: list = BBOX_LR
    _epsg: int = 2154
    _gdf: gpd.GeoDataFrame = None
    _output_path: str = None
    _output_path_shp: str = None
    _filename_shp: str = None

    collect_path = Path(os.path.join(Path(__file__).parent, 'collect'))
    physics_path = Path(os.path.join(Path(__file__).parent, 'physics'))
    meteo_path = Path(os.path.join(Path(__file__).parent, 'meteo'))
    pyqgis_path = Path(os.path.join(Path(__file__).parent, 'pyqgis'))

    geoclimate_temp_directory = f'osm_{_bbox[1]}_{_bbox[0]}_{_bbox[3]}_{_bbox[2]}'

    @classproperty
    def bbox(cls):
        """this is the bbox attribute - each subclass of Foo gets its own.
        Lookups should follow the method resolution order.
        """
        cls.geoclimate_temp_directory = (
            f'osm_{cls._bbox[1]}_{cls._bbox[0]}_{cls._bbox[3]}_{cls._bbox[2]}'
        )
        return cls._bbox

    @bbox.setter
    def bbox(cls, value):
        cls._bbox = value

    @bbox.deleter
    def bbox(cls):
        del cls._bbox

    @classproperty
    def epsg(cls):
        """this is the epsg attribute - each subclass of Foo gets its own.
        Lookups should follow the method resolution order.
        """
        return cls._epsg

    @epsg.setter
    def epsg(cls, value):
        cls._epsg = value

    @epsg.deleter
    def epsg(cls):
        del cls._epsg

    @property
    def gdf(self):
        return self._gdf

    @gdf.setter
    def gdf(self, value):
        self._gdf = value

    @property
    def output_path(self):
        return self._output_path

    @output_path.setter
    def output_path(self, value):
        self._output_path = value

    @property
    def output_path_shp(self):
        return self._output_path_shp

    @output_path_shp.setter
    def output_path_shp(self, value):
        self._output_path_shp = value

    @property
    def filename_shp(self):
        return self._filename_shp

    @filename_shp.setter
    def filename_shp(self, value):
        self._filename_shp = value

    def to_shp(self, name: str):
        try:
            from pandas.api.types import is_datetime64_any_dtype as is_datetime

            self.gdf = self.gdf[
                [
                    column
                    for column in self.gdf.columns
                    if not is_datetime(self.gdf[column])
                ]
            ]
            self.gdf = self.gdf[self.gdf.geometry.type != 'LineString']
            print(self.gdf.head(100))
        except Exception as e:
            print(f'ERROR {name} to_shp ==>', e)
            pass

        self._filename_shp = f'{name}.shp'
        self._output_path_shp = os.path.join(self._output_path, self._filename_shp)
        if os.path.exists(self._output_path_shp):
            os.remove(self._output_path_shp)
        self.gdf.to_file(filename=self._output_path_shp, driver='ESRI Shapefile')
        return self

    def to_gpkg(self, name: str):
        # Write the GeoDataFrame to a GPKG file
        self.gdf.to_file(f'{os.path.join(self.output_path, name)}.gpkg', driver='GPKG')

    @staticmethod
    def plot_scatter(
        df,
        metric_col,
        x='lng',
        y='lat',
        marker='.',
        alpha=1,
        figsize=(16, 12),
        colormap='viridis',
    ):
        df.plot.scatter(
            x=x,
            y=y,
            c=metric_col,
            title=metric_col,
            edgecolors='none',
            colormap=colormap,
            marker=marker,
            alpha=alpha,
            figsize=figsize,
        )
        plt.xticks([], [])
        plt.yticks([], [])

    @staticmethod
    def gdf_to_hex(gdf: gpd.GeoDataFrame, resolution: int = 10) -> gpd.GeoDataFrame:
        """
        https://h3geo.org/docs/core-library/restable/
        Resolution
        Average Hexagon Area (m2) Pentagon Area* (m2)
        8	737,327.598	372,048.038
        9	105,332.513	53,147.195
        10	15,047.502	7,592.318
        11	2,149.643	1,084.609
        12	307.092	154.944
        13	43.870 22.135
        :param gdf:
        :param resolution:
        :return:
        """
        # cpr_gdf = gdf.to_crs(2154)
        # 0.5m
        # buffer_length_in_meters = (0.5 * 1000) * 1.60934
        # cpr_gdf['geometry'] = cpr_gdf.geometry.buffer(buffer_length_in_meters)
        gdf_4326 = gdf.to_crs(4326)

        checkMultiPolygon = 'MultiPolygon' in list(gdf['geometry'].geom_type)
        if checkMultiPolygon:
            gdf_4326 = BasicFunctions.drop_z(gdf_4326)

        hex_gdf = gdf_4326.h3.polyfill_resample(resolution=resolution).reset_index()
        hex_gdf.rename(columns={'h3_polyfill': f'geo_h3_{resolution}'}, inplace=True)
        hex_gdf = geo_lat_lon_from_h3(hex_gdf, f'geo_h3_{resolution}')

        # hex_gdf['point'] = gpd.points_from_xy(hex_gdf['lon'], hex_gdf['lat'])
        # hex_gdf = gdf.h3.polyfill(resolution=resolution, explode=True)
        # hex_gdf = hex_gdf.h3.h3_to_geo_boundary()
        return hex_gdf

    @staticmethod
    def gen_hexagons(
        gdf: gpd.GeoDataFrame, resolution: int = 10, intersection: bool = False
    ) -> gpd.GeoDataFrame:
        """
        Converts an input multipolygon layer to H3 hexagons given a resolution.
        Parameters
        ----------
        resolution: int, 0:15
            Hexagon resolution, higher values create smaller hexagons.
        gdf: GeoDataFrame
            Input city polygons to transform into hexagons.
        Returns
        -------
        gdf_hexagons: GeoDataFrame
            Hexagon geometry GeoDataFrame (hex_id, geom).
        Examples
        --------
        :param gdf:
        :param resolution:
        :param intersection:
        """
        # class PdEncoder(json.JSONEncoder):
        #     def default(self, obj):
        #         if isinstance(obj, pd.Timestamp):
        #             return str(obj)
        #         return json.JSONEncoder.default(self, obj)

        # https://github.com/mastersigat/GeoPandas/blob/main/Prisenmain_H3_Index.ipynb
        # https://geographicdata.science/book/data/h3_grid/build_sd_h3_grid.html

        gdf = gdf.explode(index_parts=True).reset_index(drop=True)

        # print("geom_type", gdf.geom_type)
        gdf['geom_type'] = gdf.geom_type

        # gdf['geometry'] = gdf.buffer(0.1)
        cpr_gdf = gdf.to_crs(2154)
        # 0.5m
        buffer_length_in_meters = (0.5 * 1000) * 1.60934
        cpr_gdf['geometry'] = cpr_gdf.geometry.buffer(buffer_length_in_meters)

        gdf = cpr_gdf.to_crs(4326)
        gdf = gdf[gdf.geom_type == 'Polygon']

        # Obtain hexagonal ids by specifying geographic extent and hexagon resolution
        # geojson = json.loads(gdf.to_json(cls=PdEncoder))
        # print(geojson)

        h3_polygons = list()
        h3_indexes = list()

        for _, geo in gdf.iterrows():
            hexagons = h3.polyfill(
                geo['geometry'].__geo_interface__,
                res=resolution,
                geo_json_conformant=True,
            )
            for hexagon in hexagons:
                h3_polygons.append(geo_boundary_to_polygon(hexagon))
                h3_indexes.append(hexagon)

        # Create hexagon dataframe
        hex_gdf = gpd.GeoDataFrame(
            data=h3_indexes, geometry=h3_polygons, crs=4326
        ).drop_duplicates()
        hex_gdf = hex_gdf.rename(
            columns={0: 'hexid'}
        )  # Format column name for readability

        if intersection:
            # TODO https://github.com/EL-BID/urbanpy/blob/master/urbanpy/geom/geom.py
            pass
            # columns = ["nature"]
            # hex_gdf = hex_gdf.to_crs(2154)
            # polygons_ = gdf.copy().to_crs(2154)  # Preserve data state
            # polygons_['poly_area'] = polygons_.geometry.area  # Calc polygon area
            #
            # # Overlay intersection
            # overlayed = gpd.overlay(polygons_, hex_gdf, how='intersection')
            # print(overlayed.head())
            # print(overlayed.columns)
            # # Downsample indicators using proporional overlayed area w.r.t polygon area
            # area_prop = overlayed.geometry.area / overlayed['poly_area']
            # overlayed[columns] = overlayed[columns].apply(lambda col: col * area_prop)
            #
            # # Aggregate over Hex ID
            # per_hexagon_data = overlayed.groupby("hexid")[columns].sum()
            #
            # # Preserve data as GeoDataFrame
            # hex_df = pd.merge(left=per_hexagon_data, right=hex_gdf[["hexid", 'geometry']], on="hexid")
            # print(hex_df.head())
            # print(hex_df.columns)
            # hex_gdf = gpd.GeoDataFrame(hex_df[["hexid"] + columns], geometry=hex_df['geometry'], crs=hex_gdf.crs)

        #     # # Reset index to move hexagonal polygon id list to its own column
        #     hex_gdf = hex_gdf.reset_index()
        #     #
        #     # # Rename column names to allow spatial overlay operation later
        #     hex_gdf.columns = ['hexid', 'geometry']
        #     #
        #     # ### Step 3 (Final Step): spatial intersection of network and hexagonal grids
        #     #
        #     # hex_gdf = gdf.overlay(hex_gdf, how='intersection')
        #     hex_gdf = clip(gdf=hex_gdf, mask=gdf)

        return hex_gdf

    @staticmethod
    def aggregate_to_h3(gdf: gpd.GeoDataFrame, resolution: int):
        # gdf only provided for Point geometries

        return (
            gdf.to_crs(4326)
            .assign(lng=lambda x: x.geometry.x, lat=lambda x: x.geometry.y)
            .h3.geo_to_h3(resolution=resolution)
            .h3.h3_to_geo()
            .to_crs(2154)
            .assign(
                h3_easting=lambda x: x.geometry.x.astype(int),
                h3_northing=lambda x: x.geometry.y.astype(int),
            )
            .reset_index()
            .drop('geometry', axis=1)
        )

    @staticmethod
    def raster_to_h3(
        input_filename: str = 'gh_od_raster.tiff',
        output_filename: str = 'gh_od_h3.json',
    ):
        f = rasterio.open(input_filename)
        band1 = f.read(1)

        dset = {}
        for x in range(len(band1)):
            for y in range(len(band1[x])):
                lat, lng = f.xy(x, y)

                val = band1[x][y]

                hex_id = h3.geo_to_h3(
                    lat, lng, 7
                )  # swap lng and lat positions to fix layer orientiation
                dset[hex_id] = {'lat': float(lat), 'lng': float(lng), 'val': int(val)}

        json_object = json.dumps(dset, indent=4).decode('utf-8')

        with open('./' + output_filename, 'r+') as dset_f:
            dset_f.write(json_object)

    def from_geojson_to_bbox(input_filepath: str = 'path/to/file'):
        data = gpd.read_file(input_filepath)
        geom = box(
            data.bounds.minx[0],
            data.bounds.miny[0],
            data.bounds.maxx[0],
            data.bounds.maxy[0],
        )
        bbox = gpd.GeoDataFrame(geometry=[geom])
        return bbox

    def geojson_merger(path='/path/to/geojson/files'):
        """
        A commenter
        :param gejoson:
        :return:
        """
        import geojson

        geojsonFiles = glob.glob(path + '/*.geojson')

        listOfGeoJson = []
        for file in geojsonFiles:
            with open(file) as f:
                listOfGeoJson.append(geojson.load(f))
        schema = {'properties': {'features': {'mergeStrategy': 'append'}}}
        merger = Merger(schema)
        base = listOfGeoJson[0]
        for geojson in listOfGeoJson[1:]:
            merge = merger.merge(base, geojson)
            base = merge

        with open('merge.geojson', 'w') as f:
            json.dump(base, f)

    # # TODO : NE MARCHE PAS
    # def raster_to_hex(self, raster_filename: str = 'input.tiff', output_filename: str = 'output.shp',
    #                   band_name='elevation'):
    #     # Translate tif to XYZ dataframe
    #     import rioxarray as rxr
    #     df = (rxr.open_rasterio(raster_filename)
    #           .sel(band=1)
    #           .to_pandas()
    #           .stack()
    #           .reset_index()
    #           .rename(columns={'x': 'lng', 'y': 'lat', 0: band_name}))
    #
    #     print("raster_to_hex", df.head())
    #
    #     APERTURE_SIZE = 9
    #     hex_col = 'hex' + str(APERTURE_SIZE)
    #
    #     # find hexs containing the points
    #     df[hex_col] = df.apply(lambda x: h3.geo_to_h3(x.lat, x.lng, APERTURE_SIZE), 1)
    #
    #     # calculate elevation average per hex
    #     df_dem = df.groupby(hex_col)[band_name].mean().to_frame(band_name).reset_index()
    #
    #     # find center of hex for visualization
    #     df_dem['lat'] = df_dem[hex_col].apply(lambda x: h3.h3_to_geo(x)[0])
    #     df_dem['lng'] = df_dem[hex_col].apply(lambda x: h3.h3_to_geo(x)[1])
    #
    #     print("raster_to_hex", df_dem.head())
    #
    #     # plot the hexs
    #     # self.plot_scatter(df_dem, metric_col='1', marker='o', figsize=(17, 15))
    #     # plt.title('hex-grid: noise complaints')
    #     # plt.show()
    #     gdf_dem = gpd.GeoDataFrame(df_dem, geometry=gpd.points_from_xy(df_dem.lng, df_dem.lat), crs=2154)
    #     # gdf_dem.to_file(output_filename, driver='ESRI Shapefile')
    #     print("gdf_dem", gdf_dem.head())
    #     gdf_dem.plot()
    #     plt.show()
    #
    #     # self.gdf_to_hex(gdf_dem, 6).plot()
    #
    #     return

    def download_file_gitlab(
        self, host, token, project_name, branch_name, file_path, output=None
    ):
        try:
            gl = gitlab.Gitlab(url=host, private_token=token)
            pl = gl.projects.list(search=project_name)
            # print(gl.projects.list())
            project = None
            for p in pl:
                print(p.name)
                if p.name == project_name:
                    project = p
                    break
            raw_content = project.files.raw(
                file_path=file_path, ref=branch_name, streamed=False
            )
            if output:
                with open(output, 'wb') as f:
                    project.files.raw(
                        file_path=file_path,
                        ref=branch_name,
                        streamed=True,
                        action=f.write,
                    )
            return raw_content
        except Exception as e:
            print('Error:', e)
