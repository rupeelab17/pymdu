import os

import geopandas as gpd
import h3pandas
import osmnx as ox
from shapely.geometry import box

from pymdu.GeoCore import GeoCore
from pymdu.collect.GlobalVariables import TEMP_PATH
from pymdu.collect.osm.OsmCollect import OsmCollect

try:
    from osgeo import gdal, ogr
except ImportError:
    pass


class Pedestrian(GeoCore):
    """
    Class to collect the Pedestrian data
    """

    def __init__(self, filepath_shp: str = None, output_path: str = None):
        """
        Initializes the object with the given parameters.

        Args:
            filepath_shp (str): The file path to the shapefile.
            output_path (str): The output path for the processed data. If not provided, a default temporary path will be used.

        Example:
            ```python exec="true" source="tabbed-right" html="1" tabs="Source code|Plot"
            import matplotlib.pyplot as plt
            import h3pandas

            plt.clf()  # markdown-exec: hide
            import pymdu.geometric.Pedestrian as Pedestrian

            pedestrian = Pedestrian(output_path='./')
            pedestrian.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
            pedestrian.run()
            gdf = pedestrian.to_gdf()
            point_arbres = pedestrian.tree_position(
                pedestrian=pedestrian.gdf,
                resolution=11,
                height=6.0,
                type=2,
                trunk_zone=3.0,
            )
            gdf.plot(ax=plt.gca(), edgecolor='black')
            # point_arbres.plot(color='red', markersize=10)
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
        self.filepath_shp = filepath_shp
        self.output_path = output_path if output_path else TEMP_PATH

    def run_dev(self):
        if not self.filepath_shp:
            """
            way["highway"~"secondary|tertiary|residential|unclassified|service"]["sidewalk"!~"."]({{bbox}});
            """
            osm = OsmCollect(
                key='"highway"~"secondary|tertiary|residential|unclassified|service"'
            )
            self.gdf = osm.run().to_gdf()
        return self

    def run(self):
        if self.filepath_shp:
            self.gdf = gpd.read_file(self.filepath_shp, driver='ESRI Shapefile')
        else:
            envelope_bbox_polygon = box(
                self._bbox[0], self._bbox[1], self._bbox[2], self._bbox[3]
            )

            try:
                ### GRAPH ####
                # custom_filters = (
                #     '["area"!~"yes"]["highway"~"footway|service|pedestrian|cycleway"]["foot"!~"no"]["service"!~"private"]{}').format(
                #     ox.settings.default_access)
                # custom_filters = ('["highway"~"pedestrian"]')
                # print("custom_filter", custom_filters)
                # print("envelope_polygon", envelope_bbox_polygon)
                # print("envelope_polygon.length", envelope_bbox_polygon.length * 1000)
                # print("envelope_polygon.length", envelope_bbox_polygon.area)
                # G = ox.graph_from_polygon(polygon=envelope_bbox_polygon, retain_all=True, truncate_by_edge=True,
                #                           network_type='all',
                #                           simplify=False, custom_filter=custom_filters)
                #  ped = ox.graph_to_gdfs(G, nodes=False)
                ### GRAPH ####

                ped = ox.features_from_polygon(
                    polygon=envelope_bbox_polygon,
                    tags={'highway': ['footway', 'pedestrian', 'cycleway']},
                )

                ped = ped.to_crs(self._epsg)
                ped = ped.buffer(2.0)
                self.gdf = gpd.GeoDataFrame(ped, columns=['geometry'], crs='epsg:2154')
                self.gdf = self.gdf.to_crs(self._epsg)
            except Exception as e:
                print('Pedestrian error', e)
                self.gdf = gpd.GeoDataFrame([], columns=['geometry'], crs='epsg:2154')
                self.gdf = self.gdf.to_crs(self._epsg)

        return self

    def to_gdf(self) -> gpd.GeoDataFrame:
        return self.gdf

    def to_gpkg(self, name: str = 'pietons'):
        # Write the GeoDataFrame to a GPKG file
        self.gdf.to_file(f'{os.path.join(self.output_path, name)}.gpkg', driver='GPKG')

    @staticmethod
    def tree_position(
        pedestrian: gpd.GeoDataFrame,
        resolution: int = 11,
        height: float = 6.0,
        type: int = 2,
        trunk_zone: float = 3.0,
        diameter: float = 4.0,
    ) -> gpd.GeoDataFrame:
        """
        Args:
            diameter:
            height:
            type:
            trunk_zone:
            pedestrian: le geodataframe de la zone piétonne
            resolution: la résolution pour le découpage hexagonal

        Returns: geodataframe de la position des arbres

        """

        if pedestrian.empty:
            return None

        pedestrian = pedestrian.explode(ignore_index=True)
        pedestrian = pedestrian.to_crs(4326)
        # Resample to H3 cells
        print(h3pandas.__version__)
        position_arbres = pedestrian.h3.polyfill_resample(resolution)
        position_arbres['centre'] = [x.centroid for x in position_arbres['geometry']]
        point_arbres = position_arbres.copy()
        point_arbres['geometry'] = position_arbres['centre']
        point_arbres['height'] = [height for x in position_arbres['centre']]
        point_arbres['type'] = [type for x in position_arbres['centre']]
        point_arbres['trunk zone'] = [trunk_zone for x in position_arbres['centre']]
        point_arbres['diameter'] = [diameter for x in position_arbres['centre']]

        return point_arbres


if __name__ == '__main__':
    geocore = GeoCore()
    pedestrian = Pedestrian(output_path='./')
    pedestrian.bbox = [-1.152704, 46.181627, -1.139893, 46.18699]
    pedestrian = pedestrian.run()
    pedestrian_gdf = pedestrian.to_gdf()
    pedestrian.to_shp(name='ZonePietonne')

    point_arbres = pedestrian.tree_position(
        pedestrian=pedestrian.gdf,
        resolution=11,
        height=6.0,
        type=2,
        trunk_zone=3.0,
    )

    import matplotlib.pyplot as plt

    ax = pedestrian_gdf.plot(ax=plt.gca(), edgecolor='black')
    # point_arbres.plot(color='red', ax=ax, markersize=10)
    # point_arbres.plot(color='black', markersize=10, ax=ax)
    plt.show()
